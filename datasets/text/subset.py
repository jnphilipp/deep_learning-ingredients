# -*- coding: utf-8 -*-

import math
import os
import sys

from ingredients.datasets import ingredients


@ingredients.capture
def generate(DATASETS_DIR, dataset, new_set, max_vocab_size, _log,
             filters=None, clean=None):
    _log.info('Generating subset [%s: %s]' % (dataset, new_set))

    sentences = lines()
    search_list = []
    for i in range(len(sentences)):
        sentence = sentences[i].strip()
        if filters is not None:
            if filters(sentences[i]):
                if clean is not None:
                    sentences[i] = clean(sentences[i].strip())
                else:
                    sentences[i] = sentences[i].strip()
        else:
            if clean is not None:
                sentences[i] = clean(sentences[i].strip())
            else:
                sentences[i] = sentences[i].strip()
        if type(str) == sentences[i]:
            sentences[i] = re.split(r'\s+', sentences[i][0])
        vocab = set(sentences[i])
        search_list.append(i)
        sentences[i] = (len(vocab), vocab, sentence)

    selected = []
    vocab = dict()
    while True:
        min_vocab = set(vocab.keys())
        min_vocab_len = None
        min_vocab_idx = []
        subset_idx = []
        for i in search_list:
            tmp_vocab = set(sentences[i][1]).union(vocab.keys())
            if sentences[i][1].issubset(vocab.keys()):
                subset_idx.append(i)
            elif min_vocab_len is None:
                min_vocab = tmp_vocab
                min_vocab_len = len(tmp_vocab)
                min_vocab_idx = [i]
            elif len(tmp_vocab) < min_vocab_len and \
                    len(vocab) + sentences[i][0] <= max_vocab_size:
                min_vocab = tmp_vocab
                min_vocab_len = len(tmp_vocab)
                min_vocab_idx = [i]
            elif len(tmp_vocab) == min_vocab_len and min_vocab == tmp_vocab:
                min_vocab_idx.append(i)

            if len(tmp_vocab) == len(vocab) + 1:
                break

        min_vocab_idx += subset_idx
        if min_vocab_idx == []:
            break

        b = False
        for i in min_vocab_idx:
            if sentences[i][0] + len(vocab) >= max_vocab_size:
                b = True
                continue

            selected.append(i)
            for w in set(sentences[i][1]):
                if w not in vocab:
                    vocab[w] = len(vocab)
            search_list.remove(i)

        sys.stdout.write('\rvocab size: %d - sentences: %d' % (len(vocab),
                                                               len(selected)))
        if len(vocab) >= max_vocab_size or b:
            for i in search_list:
                if sentences[i][1].issubset(vocab.keys()):
                    subset_idx.append(i)
            for i in subset_idx:
                selected.append(i)
                search_list.remove(i)
            sys.stdout.write('\rvocab size: %d - sentences: %d' %
                             (len(vocab), len(selected)))
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
                f.write('%s\n' % sentences[selected[j]][2])
