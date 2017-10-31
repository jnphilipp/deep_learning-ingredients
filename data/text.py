# -*- coding: utf-8 -*-

import math
import os
import sys

from ingredients.data import ingredients


@ingredients.capture
def sentences(DATASETS_DIR, dataset, which_set, filters=None, clean=None):
    print('Loading sentences [dataset=%s - which_set=%s]' % (dataset,
                                                             which_set))
    sentences = []
    for file in os.listdir(os.path.join(DATASETS_DIR, dataset, which_set)):
        with open(os.path.join(DATASETS_DIR, dataset, which_set, file), 'r',
                  encoding='utf-8') as f:
            for line in f:
                if filters is not None:
                    if filters(line):
                        if clean is not None:
                            sentences.append(clean(line.strip()))
                        else:
                            sentences.append(line.strip())
                else:
                    if clean is not None:
                        sentences.append(clean(line.strip()))
                    else:
                        sentences.append(line.strip())
    return sentences


@ingredients.capture
def generate_subset(DATASETS_DIR, dataset, new_set, max_vocab_size,
                    filters=None, clean=None):
    sentence_list = sentences()
    print('Generating subset [dataset=%s - which_set=%s]' % (dataset, new_set))

    search_list = []
    for i in range(len(sentence_list)):
        sentence = sentence_list[i].strip()
        if filters is not None:
            if filters(sentence_list[i]):
                if clean is not None:
                    sentence_list[i] = clean(sentence_list[i].strip())
                else:
                    sentence_list[i] = sentence_list[i].strip()
        else:
            if clean is not None:
                sentence_list[i] = clean(sentence_list[i].strip())
            else:
                sentence_list[i] = sentence_list[i].strip()
        if type(str) == sentence_list[i]:
            sentence_list[i] = re.split(r'\s+', sentence_list[i][0])
        vocab = set(sentence_list[i])
        search_list.append(i)
        sentence_list[i] = (len(vocab), vocab, sentence_list[i], sentence)

    selected = []
    vocab = dict()
    while True:
        min_vocab = set(vocab.keys())
        min_vocab_len = None
        min_vocab_idx = []
        for i in search_list:
            tmp_vocab = set(sentence_list[i][1]).union(vocab.keys())
            if min_vocab_len is None:
                min_vocab = tmp_vocab
                min_vocab_len = len(tmp_vocab)
                min_vocab_idx = [i]
            if len(tmp_vocab) < min_vocab_len and \
                    len(vocab) + sentence_list[i][0] <= max_vocab_size:
                min_vocab = tmp_vocab
                min_vocab_len = len(tmp_vocab)
                min_vocab_idx = [i]
            elif len(tmp_vocab) == min_vocab_len and min_vocab == tmp_vocab:
                min_vocab_idx.append(i)

        if min_vocab_idx == []:
            break

        b = False
        for i in min_vocab_idx:
            if sentence_list[i][0] + len(vocab) >= max_vocab_size:
                b = True
                continue

            selected.append(sentence_list[i][3])
            for w in set(sentence_list[i][1]):
                if w not in vocab:
                    vocab[w] = len(vocab)
            search_list.remove(i)

        sys.stdout.write('\rvocab size: %d - sentences: %d' % (len(vocab),
                                                               len(selected)))
        if len(vocab) >= max_vocab_size or b:
            break
    print('')

    os.makedirs(os.path.join(DATASETS_DIR, dataset, new_set))
    for i in range(math.ceil(len(selected) / 100000)):
        with open(os.path.join(DATASETS_DIR, dataset, new_set, '%03d.txt' % i),
                  'w', encoding='utf-8') as f:
            start = i * 100000
            end = (i + 1) * 100000
            if end >= len(selected):
                end = len(selected)
            for j in range(start, end):
                f.write('%s\n' % selected[j])
