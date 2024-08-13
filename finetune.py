from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback
import torch
import json
import time
from datasets import load_from_disk

import config
from utils import load_model


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = 0
        for feature in features:
            max_length = max(max_length, len(feature['input_ids']))
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for feature in features:
            batch['input_ids'].append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'])))
            batch['attention_mask'].append(feature['attention_mask'] + [0] * (max_length - len(feature['attention_mask'])))
            batch['labels'].append(feature['input_ids'] + [-100] * (max_length - len(feature['input_ids'])))
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch


class LossCallback(TrainerCallback):
    def __init__(self, log_list):
        self.log_list = log_list
    
    def on_train_step_end(self, args, state, control, logs=None, **kwargs):
        step_log = logs
        self.loss_list.append(step_log)
        print(f"Step {state.global_step}: Loss = {step_log['loss']}")


def training(training_pairs):

    model = load_model()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def tokenize(example):
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\n{example['question']}<|im_end|>\n\
<|im_start|>assistant\n{example['answer']}<|im_end|>"
        token = tokenizer(prompt, return_tensors='pt')
        token['input_ids'] = token['input_ids'].squeeze(0)
        token['attention_mask'] = token['attention_mask'].squeeze(0)
        token['labels'] = token['input_ids']
        return token

    training_pairs = training_pairs.map(tokenize)

    training_args = TrainingArguments(
        output_dir="./results/ckpts",
        num_train_epochs=4,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=1,
        deepspeed='ds_z2_config.json',
    )

    log_list = []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_pairs,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer),
        callbacks=[LossCallback(log_list)]
    )
    trainer.train()

    print(trainer.state.log_history)
    with open('results/log.json', 'w') as f:
        json.dump(trainer.state.log_history, f)

    model.save_pretrained('results/LoRA')


if __name__ == '__main__':
    begin_time = time.time()
    training_pairs = load_from_disk('results/sft_data')
    print(training_pairs)
    training(training_pairs)
    end_time = time.time()
    print(f"Total sft time: {end_time - begin_time}")
