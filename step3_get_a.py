import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from datasets import Dataset
import json

from api_client import get_response
import config


class ContextQA(dspy.Signature):
    """给定一个上下文参考和一个问题，返回答案，你必须遵循以下原则。
原则
1. 你必须借助上下文参考准确无误地回答问题。
2. 回答时需要仿照人类回答问题的方式。
3. 请不要出现违反道德的答案。
    """
    context = dspy.InputField(desc="上下文参考")
    question = dspy.InputField(desc="问题")
    answer = dspy.OutputField(desc="你的答案")

# For English version:
# class ContextQA(dspy.Signature):
#     """Given a contextual reference and a question, to return the answer, you must follow the following principles.
# Principles
# 1. You must use contextual references to answer questions accurately and without error.
# 2. When answering, it is necessary to imitate the way humans answer questions.
# 3. Please do not provide answers that violate ethics.
#     """
#     context = dspy.InputField(desc="Context reference")
#     question = dspy.InputField(desc="question")
#     answer = dspy.OutputField(desc="Your answer")


class GetAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ContextQA)
    
    def forward(self, question, context):
        return self.prog(question=question, context=context)


def get_a(train_set, q_dataset):

    lm = dspy.OpenAI(
        model=config.model_name_or_path,
        api_base="http://localhost:8000/v1/",
        api_key='EMPTY',
        stop="---",
    )
    dspy.settings.configure(lm=lm)

    new_train_set = []
    for data in train_set:
        new_train_set.append(dspy.Example(context=data['context'], 
                                    question=data['question'],
                                    answer=data['answer']).with_inputs("context", "question"))
    train_set = new_train_set

    new_all_set = []
    for data in q_dataset:
        new_all_set.append(dspy.Example(context=data['context'],
                                        question=data['question']).with_inputs("context", "question"))
    q_dataset = new_all_set

    def gold_metric(example, pred, trace=None):  # Let the model self-evaluate its own output to optimize few shot examples

        prompt = f"""根据相关上下文和参考答案，确定另一个新预测的答案是否准确回答了问题，返回是或否。
[相关上下文]: {example.context}
[问题]: {example.question}
[参考答案]: {example.answer}
[预测答案]: {pred.answer}
[是/否]: """


# For English version: 
#         prompt = f"""Determine whether another newly predicted answer accurately answered the question based on the context reference and reference answer, and return yes or no.
# [Context Reference]: {example.context}
# [Question]: {example.question}
# [Reference Answer]: {example.answer}
# [Predicted Answer]: {pred.answer}
# [yes/no]: """

        score = get_response(prompt)
        score = score.strip()
        print('SCORE: ', score)
        print('END')
        if '是' in score.lower() or 'yes' in score.lower():
            return True
        if '否' in score.lower() or 'no' in score.lower():
            return False
        assert False

    dspy_config = dict(max_bootstrapped_demos=3, max_labeled_demos=3)
    teleprompter = BootstrapFewShot(metric=gold_metric, **dspy_config)
    optimized = teleprompter.compile(GetAModule(), trainset=train_set)  # 训练

    print('\n\n\n\n\n\n\n\n')
    lm.inspect_history(n=1)

    def dev_metric(example, pred, trace=None):
        return 1
    
    evaluate = Evaluate(devset=q_dataset, metric=dev_metric, num_threads=8, display_progress=True,
                        display_table=0, return_outputs=True)

    _, outputs = evaluate(optimized)  # Inference

    qa_dataset = {
        "context": [],
        "question": [],
        "answer": []
    }

    for output in outputs:
        example, pred, score = output
        assert score == 1
        if score != 1:
            continue
        qa_dataset["context"].append(example.context)
        qa_dataset["question"].append(example.question)
        qa_dataset["answer"].append(pred.answer)

    qa_dataset = Dataset.from_dict(qa_dataset)
    print(qa_dataset)
    return qa_dataset


if __name__ == '__main__':
    train_set = {
        "context": [],
        "question": [],
        "answer": []
    }
    with open('qa_examples_zh.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_set["context"].append(data['context'])
            train_set["question"].append(data['question'])
            train_set["answer"].append(data['answer'])
    train_set = Dataset.from_dict(train_set)
    print('train_set: ', train_set)

    with open('results/filtered_queries.json', 'r') as f:
        queries = json.load(f)
    q_dataset = {
        "context": [],
        "question": [],
    }
    for id, lst in queries.items():
        for context, query in lst:
            q_dataset["context"].append(context)
            q_dataset["question"].append(query)
    print('q_dataset: ', q_dataset)
    
    qa_data = get_a(train_set, q_dataset)
    qa_data.save_to_disk('results/sft_data')
    print(qa_data)
