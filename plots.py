# -*- coding: utf-8 -*-
# Copyright (C) 2019 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of deep_learning-ingredients.
#
# deep_learning-ingredients is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# deep_learning-ingredients is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with deep_learning-ingredients. If not, see
# <http://www.gnu.org/licenses/>.

import numpy as np
import os

from sacred import Ingredient
from sacred.run import Run


ingredient = Ingredient('plots')


@ingredient.capture
def scatter(name: str, data: list, legend: bool = True, xlim: int = None,
            ylim: int = None, _run: Run = None):
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
def lines(name: str, x1_data: dict, x2_data: dict = None, legend: bool = True,
          _run: Run = None):
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
    if x2_data:
        for row in x2_data['lines']:
            lines.append(ax2.plot(row['x'], row['y'], label=row['label']))

    if 'xlabel' in x1_data:
        ax1.set_xlabel(x1_data['xlabel'])
    if 'ylabel' in x1_data:
        ax1.set_ylabel(x1_data['ylabel'])
    if x2_data and 'xlabel' in x2_data:
        ax2.set_xlabel(x2_data['xlabel'])
    if x2_data and 'ylabel' in x2_data:
        ax2.set_ylabel(x2_data['ylabel'])

    ax1.legend([l for ls in lines for l in ls],
               [l.get_label() for ls in lines for l in ls],
               loc=9,
               bbox_to_anchor=(0.5, -0.07),
               ncol=4)
    fig.savefig(os.path.join(_run.observers[0].run_dir, '%s.png' % name),
                format='png', bbox_inches='tight')
    plt.close()
