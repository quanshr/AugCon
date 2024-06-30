from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch.nn as nn
import torch
from typing import Optional
import os
import json
from utils import calc_rouge, gen_count
from datasets import load_from_disk, Dataset
from peft import get_peft_model, prepare_model_for_kbit_training

import config
from load_data import load_data


class Scorer(nn.Module):
    """
    Train a scorer to rate the quality of the problem, which is structured as a reward model and trained through contrastive learning 
    using positive and negative examples composed of optimal and suboptimal prompts

    """

    def __init__(self):
        super().__init__()
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        self.config = self.model.config
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, config.lora_config)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,
                input_ids: torch.LongTensor, # 2B * L
                attention_mask: Optional[torch.Tensor] = None,
                labels: torch.FloatTensor = None) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        if attention_mask is None:
            last_hidden_states = last_hidden_states[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
            last_hidden_states = last_hidden_states.gather(1, last_index.view(-1, 1, 1).expand(-1, 1, last_hidden_states.size(-1))).squeeze(1)
        scores = self.value_head(last_hidden_states)
        scores = self.sigmoid(scores)
        
        if labels is None:
            return scores
        
        assert input_ids.shape[0] % 2 == 0
        B = input_ids.shape[0] // 2
        loss = -scores[:B].mean() + scores[B:].mean()
        return loss, scores


class CustomDataCollator:  # This is mainly for padding and formatting preprocessing
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = 0
        for feature in features:
            max_length = max(max_length, len(feature['input_ids'][0]))
        batch = {'input_ids': [], 'attention_mask': []}
        for k in range(len(features[0]['input_ids'])):
            for feature in features:
                batch['input_ids'].append(feature['input_ids'][k] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'][k])))
                batch['attention_mask'].append(feature['attention_mask'][k] + [0] * (max_length - len(feature['attention_mask'][k])))
        if len(features[0]['input_ids']) == 2:
            batch['labels'] = [[1, 2] for _ in range(len(features))]
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch


def train(training_pairs):

    scorer = Scorer()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def tokenize(example):
        prompts = [f"Context: {example['context']}\n\nQuestion: {example['pos_query']}\n\n", 
                   f"Context: {example['context']}\n\nQuestion: {example['neg_query']}\n\n"]
        token = tokenizer(prompts, padding=True, return_tensors='pt')
        return token

    training_pairs = training_pairs.map(tokenize)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=4, # Reduced batch size for demonstration purposes
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        deepspeed='ds_z2_config.json',
    )

    trainer = Trainer(
        model=scorer,
        args=training_args,
        train_dataset=training_pairs,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer),
    )

    print(f'Start training, {len(training_pairs)} training pairs in total.')
    trainer.train()
    print('Finish training')

    return scorer


def score_and_rank(scorer, queries):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    dataset = {
        'id': [],
        'context': [],
        'query': [],
    }
    for id, lst in queries.items():
        for sample in lst:
            dataset['id'].append(id)
            dataset['context'].append(sample[0])
            dataset['query'].append(sample[1])
    dataset = Dataset.from_dict(dataset)
    print('Start scoring and ranking: ', dataset)
    def tokenize(example):
        prompts = [f"Context: {example['context']}\n\nQuestion: {example['query']}\n\n"]
        token = tokenizer(prompts, padding=True, return_tensors='pt')
        return token

    token_dataset = dataset.map(tokenize)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=4,
        deepspeed='ds_z3_config.json',  # Only ZerO-3 can be used at inference time
    )
    trainer = Trainer(
        model=scorer,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer),
    )
    outputs = trainer.predict(test_dataset=token_dataset).predictions
    outputs = [x[0] for x in outputs]
    dataset = dataset.add_column('score', outputs)
    dataset = dataset.sort('score', reverse=True)

    ranked_queries = {id: [] for id in queries.keys()}
    for sample in dataset:
        ranked_queries[sample['id']].append((sample['context'], sample['query']))
    print('Finish scoring and ranking')
    print('ranked_queries: ', ranked_queries)
    return ranked_queries


def filter_queries(ranked_queries):  # Filter after sorting
    corpus = load_data()
    corpus = corpus
    filtered_queries = {}
    for id, lst in ranked_queries.items():
        new_list = []
        for context, query in lst:
            ok = True
            for _, query_ref in new_list:
                if calc_rouge(query, query_ref)['f'] > config.rouge_thres:  # Directly discard items with high similarity
                    ok = False
                    break
            if ok:
                new_list.append((context, query))
                if len(new_list) >= gen_count(corpus['context'][int(id)]):
                    break
        filtered_queries[id] = new_list
    print('Finish filtering')
    print('filtered_queries: ', filtered_queries)
    return filtered_queries


if __name__ == '__main__':  # Training -> Scoring -> Ranking -> Filtering
    training_pairs = load_from_disk('results/scorer_pairs')
    scorer = train(training_pairs)
    with open('results/queries.json', 'r') as f:
        queries = json.load(f)
    ranked_queries = score_and_rank(scorer, queries)
    filtered_queries = filter_queries(ranked_queries)
    with open('results/filtered_queries.json', 'w') as f:
        json.dump(filtered_queries, f)
