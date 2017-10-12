# -*- coding: utf-8 -*-

import json
import os

from ingredients.experiment import ingredients


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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.suptitle(name, family='monospace', fontsize=20)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    lines = []
    history = load(path, name)
    for k in history.keys():
        if 'loss' in k:
            lines.append(ax1.plot(range(len(history[k])), history[k], label=k))
            next(ax2._get_lines.prop_cycler)['color']
        else:
            lines.append(ax2.plot(range(len(history[k])), history[k], label=k))
            next(ax1._get_lines.prop_cycler)['color']

    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax1.legend([l for ls in lines for l in ls],
               [l.get_label() for ls in lines for l in ls],
               loc=9,
               bbox_to_anchor=(0.5, -0.07),
               ncol=4)
    fig.savefig(os.path.join(path, '%s.png' % name),
                format='png',
                bbox_inches='tight')
    plt.close()
