# -*- coding: utf-8 -*-

import json
import os

from ingredients.experiments import ingredients
from ingredients.experiments import plots


@ingredients.capture
def load(path, name):
    with open(os.path.join(path, '%s.json' % name), 'r') as f:
        return json.loads(f.read())


@ingredients.capture
def save(path, name, history):
    with open(os.path.join(path, '%s.json' % name), 'w', encoding='utf8') as f:
        f.write(json.dumps(history, indent=4))
        f.write('\n')


@ingredients.capture
def plot(path, name='train_history'):
    history = load(path, name)

    x1_data = {'ylabel': 'loss', 'xlabel': 'epoch', 'lines': []}
    x2_data = {'ylabel': 'acc', 'lines': []}
    for k in history.keys():
        if 'loss' in k:
            x1_data['lines'].append({
                'x': range(len(history[k])),
                'y': history[k],
                'label': k
            })
        else:
            x2_data['lines'].append({
                'x': range(len(history[k])),
                'y': history[k],
                'label': k
            })

    plots.lines(path, name, x1_data, x2_data)
