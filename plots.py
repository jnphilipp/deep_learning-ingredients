# -*- coding: utf-8 -*-

import numpy as np
import os

from sacred import Ingredient


ingredient = Ingredient('plots')


@ingredient.capture
def scatter(name, data, legend=True, xlim=None, ylim=None, _run=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm
    import matplotlib.pyplot as plt

    colors = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(data))))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.suptitle(name, family='monospace', fontsize=20)
    ax = fig.add_subplot(111)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    for i, row in enumerate(data):
        s = row['s'] if 's' in row else None
        color = row['color'] if 'color' in row else next(colors)
        label = row['label'] if 'label' in row else None
        alpha = row['alpha'] if 'alpha' in row else None
        marker = row['marker'] if 'marker' in row else 'o'
        ax.scatter(row['x'], row['y'], s, color=color, marker=marker,
                   alpha=alpha, label=label)

    if legend:
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.07), ncol=4)
    fig.savefig(os.path.join(_run.observers[0].run_dir, '%s.png' % name),
                format='png', bbox_inches='tight')
    plt.close()


@ingredient.capture
def lines(name, x1_data, x2_data=None, legend=True, _run=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.suptitle(name, family='monospace', fontsize=20)
    ax1 = fig.add_subplot(111)
    if x2_data:
        ax2 = ax1.twinx()

    lines = []
    for row in x1_data['lines']:
        lines.append(ax1.plot(row['x'], row['y'], label=row['label']))
        if x2_data:
            next(ax2._get_lines.prop_cycler)['color']
    for row in x2_data['lines']:
        lines.append(ax2.plot(row['x'], row['y'], label=row['label']))

    if 'xlabel' in x1_data:
        ax1.set_xlabel(x1_data['xlabel'])
    if 'ylabel' in x1_data:
        ax1.set_ylabel(x1_data['ylabel'])
    if 'xlabel' in x2_data:
        ax2.set_xlabel(x2_data['xlabel'])
    if 'ylabel' in x2_data:
        ax2.set_ylabel(x2_data['ylabel'])

    ax1.legend([l for ls in lines for l in ls],
               [l.get_label() for ls in lines for l in ls],
               loc=9,
               bbox_to_anchor=(0.5, -0.07),
               ncol=4)
    fig.savefig(os.path.join(_run.observers[0].run_dir, '%s.png' % name),
                format='png', bbox_inches='tight')
    plt.close()
