# -*- coding: utf-8 -*-

import json
import os

from sacred import Ingredient

from . import plots as plots_ingredient

ingredient = Ingredient('history', ingredients=[plots_ingredient.ingredient])


@ingredient.capture
def load(name, _run):
    path = os.path.join(_run.observers[0].run_dir, '%s.json' % name)
    with open(path, 'r') as f:
        return json.loads(f.read())


@ingredient.capture
def save(name, history, _log, _run):
    _log.info('Save train history [%s]' % name)
    path = os.path.join(_run.observers[0].run_dir, '%s.json' % name)
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(history, indent=4))
        f.write('\n')


@ingredient.command
def plot(name='train_history', _log=None):
    _log.info('Plot %s' % name)

    history = load(name)
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

    plots_ingredient.lines(name, x1_data, x2_data)
