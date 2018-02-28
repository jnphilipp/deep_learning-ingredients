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

    entities = {}
    input_len = 0
    outputs = {'intent': ['null'], 'language': ['null']}
    texts = []
    y = {'intent': [], 'language': []}
    for example in data['common_examples']:
        text = fclean(example['text'], vocab) if fclean else example['text']
        input_len = max(input_len, len(text))
        texts.append([int(s.translate(trans)) for s in '%s^' % text])

        if example['intent'] not in outputs['intent']:
            outputs['intent'].append(example['intent'])
        if example['language'] not in outputs['language']:
            outputs['language'].append(example['language'])

        for entity in example['entities']:
            entity = entity['entity']
            if entity['parent']:
                name = entity['parent']
                child = entity['name']
            else:
                name = entity['name']
                child = None

            if name not in entities:
                entities[name] = []
                outputs[name] = ['null'] if child else []
                y[name] = []

            if child and child not in entities[name]:
                entities[name].append(child)
                outputs[name].append(child)

        y['intent'].append(outputs['intent'].index(example['intent']))
        y['language'].append(outputs['language'].index(example['language']))

    for example in data['common_examples']:
        names = [e['entity']['parent'] if e['entity']['parent']
                 else e['entity']['name'] for e in example['entities']]
        children = [e['entity']['name'] for e in example['entities']
                    if e['entity']['parent']]

        for k, v in entities.items():
            value = 0
            for entity in example['entities']:
                name = entity['entity']['name']
                parent = entity['entity']['parent']
                if k == name and len(v) == 0:
                    value = 1
                elif k == parent and len(v) > 0:
                    value = outputs[parent].index(name)
            y[k].append(value)

    X = np.zeros((len(texts), input_len + 1))
    for i, text in enumerate(texts):
        X[i][0:len(text)] = text

    for k in y.keys():
        if len(outputs[k]) > 0:
            y[k] = np_utils.to_categorical(y[k], len(outputs[k]))
        else:
            y[k] = np.asarray(y[k]).reshape((len(X), 1))

    to_del = []
    for k, v in outputs.items():
        if len(v) == 0:
            to_del.append(k)
    for i in to_del:
        del outputs[i]

    outputs['entities'] = entities
    outputs['outputs'] = list(y.keys())
    return X, y, len(vocab), outputs
