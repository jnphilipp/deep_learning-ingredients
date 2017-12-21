# -*- coding: utf-8 -*-

import json
import numpy as np
import os

from ingredients.data import ingredients
from keras.utils import np_utils


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, vocab, fclean=None):
    trans = str.maketrans(dict((s, str(i)) for i, s in enumerate(vocab)))

    data = {}
    with open(os.path.join(DATASETS_DIR, dataset, '%s.json' % which_set),
              'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    entities = []
    y_entities = {}
    for example in data['common_examples']:
        for entity in example['entities']:
            if entity['entity'] not in entities:
                entities.append(entity['entity'])
                y_entities[entity['entity']] = []

    input_len = 0
    intents = []
    languages = []
    texts = []
    y_intent = []
    y_language = []
    for example in data['common_examples']:
        text = fclean(example['text'], vocab) if fclean else example['text']
        input_len = max(input_len, len(text))
        texts.append([int(s.translate(trans)) for s in '%s^' % text])

        if example['intent'] not in intents:
            intents.append(example['intent'])
        if example['language'] not in languages:
            languages.append(example['language'])

        y_intent.append(intents.index(example['intent']))
        y_language.append(languages.index(example['language']))

        for entity in entities:
            if True in [entity == e['entity'] for e in example['entities']]:
                y_entities[entity].append(1)
            else:
                y_entities[entity].append(0)

    X = np.zeros((len(texts), input_len + 1))
    for i, text in enumerate(texts):
        X[i][0:len(text)] = text

    y = {
        'intent': np_utils.to_categorical(y_intent, len(intents)),
        'language': np_utils.to_categorical(y_language, len(languages)),
    }
    for k, v in y_entities.items():
        y[k] = np.asarray(v).reshape((len(X), 1))
    return X, y, len(vocab), {
        'intents': intents,
        'languages': languages,
        'entities': entities,
        'outputs': list(y.keys())
    }
