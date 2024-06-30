import re

import prompt
import config
from api_client import get_completion


def context_split_tree(context, data=None, id=None, lock=None, template=prompt.instruct_CST):
    """
    Given the context as initial root, recursively build the context-split-tree, and append 
    the derived context-query pairs to the data dict.

    Parameters:
        context (str): The initial context.
        data (dict): The list of context-query pairs, in format: {'id': [(context, question)]}.
            if data is None, just return a derived context-query pair and don't split further.
        id (int): The id of the initial context, it is used to group the pairs for refining.
        lock (threading.Lock): The lock to synchronize access to the data list.
        template (str): The template for the instruction.
    Returns:
        None if data is not None(The data dict is modified in-place), 
            else return a derived context-query pair.
    """
    if len(context) <= config.min_length:
        return
    instr = template.format(context=context)
    for _ in range(4):
        completion = get_completion(instr).text
        pattern = r"(.*)\s*Context 1: (.*)\s*Context 2: (.*)\s*"
        match = re.search(pattern, completion)
        if match:
            break
    if match:
        question = match.group(1).strip()
        context1 = match.group(2).strip()
        context2 = match.group(3).strip()
        if len(question) < 5:
            return
        if data is None:
            return (context, question)
        if lock is None:
            data[id].append((context, question))
        else:
            with lock:
                data[id].append((context, question))
        if len(context1) >= len(context) or len(context2) >= len(context):
                return
        context_split_tree(match.group(2), data, id)
        context_split_tree(match.group(3), data, id)
