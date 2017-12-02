# -*- coding: utf-8 -*-

import json
import numpy as np
import os

from ingredients.data import ingredients
from keras.utils import np_utils


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, vocab):
    data = {}
    with open(os.path.join(DATASETS_DIR, dataset, '%s.json' % which_set),
              'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    vocab = dict((s, str(i)) for i, s in enumerate(vocab))
    trans = str.maketrans(vocab)

    texts = []
    input_len = 0
    intents = []
    y_intents = []
    y_languages = []
    languages = []
    for e in data['common_examples']:
        input_len = max(input_len, len(e['text']))
        texts.append([int(s.translate(trans)) for s in '%s^' % e['text']])

        if e['intent'] not in intents:
            intents.append(e['intent'])
        if e['language'] not in languages:
            languages.append(e['language'])

        y_intents.append(intents.index(e['intent']))
        y_languages.append(languages.index(e['language']))

    X = np.zeros((len(texts), input_len + 1))
    for i, text in enumerate(texts):
        X[i][0:len(text)] = text

    y_intents = np_utils.to_categorical(y_intents, len(intents))
    y_languages = np_utils.to_categorical(y_languages, len(languages))

    return X, [y_intents, y_languages], len(vocab), [intents, languages]
