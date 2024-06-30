from datasets import DatasetDict, Dataset
import itertools
import re

import config


def split_text_into_chunks(context, max_length=config.max_length):
    """
    Divide the text into paragraphs that do not exceed max_length characters, 
        and try to divide them into complete sentences as much as possible.
    """
    sentences = re.split('([。？！]+)', context)
    
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2)]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def load_data():
    raw_corpus = DatasetDict.load_from_disk('DailyM')['train']
    corpus = []
    for article in raw_corpus['article']:
        corpus.extend(article.split('\n'))
    corpus = [context.strip() for context in corpus]
    corpus = [context for context in corpus if len(context) > 100]

    corpus = list(itertools.chain.from_iterable(split_text_into_chunks(context) for context in corpus))

    corpus = Dataset.from_dict({'context': corpus})
    print('finish loading corpus: ')
    print(corpus)
    return corpus

if __name__ == '__main__':

    load_data()