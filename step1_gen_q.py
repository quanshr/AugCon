import time
import concurrent.futures
from threading import Lock
from tqdm import tqdm
from datasets import Dataset
import random
import json
import os

from step1_cst import context_split_tree
from utils import gen_count
import prompt
from load_data import load_data


def gen_q(context, data, id, lock):
    """
    Due to the unnecessary time consumption caused by frequent switching between Scorer and LLM, 
        we generate an additional excess of data for filtering purposes
    """

    while len(data[id]) < gen_count(context) * 1.15: 
        context_split_tree(context, data, id, lock)


def gen_q_pool(corpus):
    data = {id: [] for id in range(len(corpus['context']))}
    lock = Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(gen_q, context, data, id, lock) \
                   for id, context in enumerate(corpus['context'])]
        
        for count, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(corpus['context']))):
            pass
    return data


def gen_pair(context, pos_query, pairs, lock, template):
    _, neg_query = context_split_tree(context, None, None, None, template)
    with lock:
        pairs.append((context, pos_query, neg_query))


def mk_scorer_pairs(data, count=500):
    pairs = []
    all_data = []
    for _, lst in data.items():
        all_data.extend(lst)
    print(f'all queries count: {len(all_data)}')
    random.shuffle(all_data)
    pairs = []
    lock = Lock()
    count = min(count, len(all_data) // 3)
    for type, neg_prompt in enumerate(prompt.neg_instruct_CST_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(gen_pair, context, question, pairs, lock, neg_prompt) \
                    for context, question in all_data[count*type:count*(type+1)]]
            for id, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(corpus['context']))):
                pass
    pairs = {
        'context': [pair[0] for pair in pairs],
        'pos_query': [pair[1] for pair in pairs],
        'neg_query': [pair[2] for pair in pairs]
    }
    pairs = Dataset.from_dict(pairs)
    return pairs
    

if __name__ == '__main__':
    print('begin')
    begin_time = time.time()

    corpus = load_data()
    corpus = corpus

    data = gen_q_pool(corpus)  # in format: {id: [(context, question)]}
    end_time1 = time.time()
    print(f'generate queries time: {end_time1 - begin_time}s')
    os.makedirs('results', exist_ok=True)
    with open('results/queries.json', 'w') as f:
        json.dump(data, f)

    pairs = mk_scorer_pairs(data)  # in Huggging Dataset format
    print(pairs)
    pairs.save_to_disk('results/scorer_pairs')

    end_time2 = time.time()
    print(f'generate scorer training pairs time: {end_time2 - end_time1}s')
