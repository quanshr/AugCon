import concurrent.futures
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from api_client import get_response


def gen_output_pool(queries):
    outputs = {'question': queries['question'], 'answer': ['']*len(queries['question'])}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_response, query) for query in queries['question']]
        
        for id, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(queries['question']))):
            outputs['answer'][id] = future.result()

    outputs = Dataset.from_dict(outputs)
    print('outputs: ', outputs)
    return outputs


if __name__ == '__main__':
    test_queries = DatasetDict.load_from_disk('DailyM')['test']
    outputs = gen_output_pool(test_queries)
    outputs.save_to_disk('results/outputs')
