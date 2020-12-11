import os
import sys
import json
import bcolz
import itertools
import numpy as np
sys.path.append(os.getcwd())
from scipy.stats import entropy
from collections import Counter, defaultdict

import utils.data as data
import utils.utils as utils
import utils.config as config


def _get_file_(train=False, val=False, test=False, question=False, answer=False):
    """ Get the correct question or answer file."""
    _file = utils.path_for(train=train, val=val, test=test, 
                            question=question, answer=answer)
    with open(_file, 'r') as fd:
        _object = json.load(fd)
    return _object


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = iterable if top_k else itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens)}
    return vocab


def extract_type(answers_dset, ans2label):
    """ Extract answer distribution for each question type. """
    qt_dict = defaultdict(list)
    for ans_entry in answers_dset:
        qt = ans_entry['question_type']
        ans_idxs = []
        for ans in ans_entry['answers']:
            ans = data.process_answers(ans['answer'])
            ans_idx = ans2label.get(ans, None)
            if ans_idx:
                ans_idxs.append(ans_idx)
        qt_dict[qt].extend(ans_idxs) # counting later

    # count answers for each question type
    for qt in qt_dict:
        ans_num_dict = Counter(qt_dict[qt])
        ans_num_dict = {k: v
            for k, v in ans_num_dict.items() if v >= 50}
        total_num = sum(ans_num_dict.values())
        for ans, ans_num in ans_num_dict.items():
            ans_num_dict[ans] = float(ans_num) / total_num

        values = np.array(list(ans_num_dict.values()), dtype=np.float32)
        if entropy(values + 1e-6, base=2) >= config.entropy:
            qt_dict[qt] = {k: 0.0 for k in ans_num_dict}
        else:
            qt_dict[qt] = ans_num_dict
    return qt_dict


def filter_glove(question_vocab):
    """ Filtering glove file and reshape the glove feature file so that the 
        embedding features are all about the current question vocabulary.
    """
    glove_file = os.path.join(config.glove_path, 'glove.6B.300d.txt')
    glove_weights = {}
    glove_weights_filtered = bcolz.carray(np.zeros(1))

    with open(glove_file, 'r') as g:
        for line in g:
            split = line.split()
            word = split[0]
            embedding = np.array([float(val) for val in split[1:]])
            glove_weights[word] = embedding

    # find words in glove which are from the current vocabulary
    for word in question_vocab:
        glove_weights_filtered.append(glove_weights.get(
                                                word, np.zeros(300)))
       
    embeddings = bcolz.carray(
        glove_weights_filtered[1:].reshape(len(question_vocab), 300), # padding
        rootdir=config.glove_path_filtered,
        mode='w')
    embeddings.flush()


def main():
    # For question processing, we aim to use the pre-trained glove embedding
    # vectors, thus all questions are processed for filtering glove.
    questions_train = _get_file_(train=True, question=True)
    questions_val = _get_file_(val=True, question=True)

    if not config.cp_data:
        questions_train = questions_train['questions']
        questions_val = questions_val['questions']

    questions_train = list(data.prepare_questions(questions_train))
    questions_val = list(data.prepare_questions(questions_val))

    questions = questions_train + questions_val

    if not config.cp_data:
        questions_test = _get_file_(test=True, question=True)
        questions_test = questions_test['questions']
        questions_test = list(data.prepare_questions(questions_test))
        questions += questions_test

    # process answers subject to fair training
    if config.train_set == 'train' and not config.cp_data:
        answers = _get_file_(train=True, answer=True)
        answers = answers['annotations']
        answers = list(data.prepare_mul_answers(answers))
    else: # train+val
        answers = []
        for train in [True, False]:
            ans = _get_file_(train=train, val=not train, answer=True)
            if not config.cp_data:
                ans = ans['annotations']
            answers += list(data.prepare_mul_answers(ans))

    answer_vocab = extract_vocab(answers, top_k=config.max_answers)
    question_vocab = extract_vocab(questions, start=0)
    filter_glove(question_vocab)

    train_answers = _get_file_(train=True, answer=True)
    qt_dict = extract_type(train_answers, answer_vocab)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
        'type': qt_dict
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
