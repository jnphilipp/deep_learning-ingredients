# -*- coding: utf-8 -*-

import json
import numpy as np
import os

from ingredients.data import ingredients
from keras.utils import np_utils


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, vocab, fclean=None, y_null=False):
    trans = str.maketrans(dict((s, str(i)) for i, s in enumerate(vocab)))

    data = {}
    with open(os.path.join(DATASETS_DIR, dataset, '%s.json' % which_set),
              'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    entities = {}
    texts = {'nlu': []}
    y = {'nlu': {'intent': [], 'language': []}}
    for example in data['common_examples']:
        for entity in example['entities']:
            entity = entity['entity']
            if entity['parent']:
                name = entity['parent']
                child = entity['name']
                texts[name] = []
            else:
                name = entity['name']
                child = None

            if name not in entities:
                entities[name] = None
                y['nlu'][name] = []

            if child and (entities[name] is None or child not in entities[name]):
                if entities[name] is None:
                    entities[name] = ['null']
                entities[name].append(child)

    input_len = 0
    intents = ['null'] if y_null else []
    languages = ['null'] if y_null else []
    for example in data['common_examples']:
        text = fclean(example['text'], vocab) if fclean else example['text']
        input_len = max(input_len, len(text))
        texts['nlu'].append([int(s.translate(trans)) for s in '%s^' % text])

        if example['intent'] not in intents:
            intents.append(example['intent'])
        if example['language'] not in languages:
            languages.append(example['language'])

        y['nlu']['intent'].append(intents.index(example['intent']))
        y['nlu']['language'].append(languages.index(example['language']))

        names = [e['entity']['parent'] if e['entity']['parent']
                 else e['entity']['name'] for e in example['entities']]

        for entity in entities:
            if True in [entity == n for n in names]:
                y['nlu'][entity].append(1)
            else:
                y['nlu'][entity].append(0)

        for k in texts.keys():
            if k == 'nlu':
                continue

            for entity in example['entities']:
                if entity['entity']['parent'] is None:
                    continue
                elif entity['entity']['parent'] != k:
                    continue

                if k not in y:
                    y[k] = []

                texts[k].append(texts['nlu'][-1])
                y[k].append(entities[entity['entity']['parent']].index(
                    entity['entity']['name'])
                )

    X = {k: np.zeros((len(v), input_len + 1)) for k, v in texts.items()}
    for k in texts.keys():
        for i, text in enumerate(texts[k]):
            X[k][i][0:len(text)] = text

    for k in y.keys():
        if k == 'nlu':
            y[k]['intent'] = np_utils.to_categorical(y[k]['intent'], len(intents))
            y[k]['language'] = np_utils.to_categorical(y[k]['language'], len(languages))
            for e, v in y[k].items():
                if e == 'intent' or e == 'language':
                    continue
                y[k][e] = np.asarray(v).reshape((len(X['nlu']), 1))
        else:
            y[k] = np_utils.to_categorical(y[k], len(entities[k]))

    return X, y, len(vocab), {
        'intent': intents,
        'language': languages,
        'entities': entities,
        'outputs': list(y['nlu'].keys())
    }
