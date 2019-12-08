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

import json
import os

from logging import Logger
from sacred import Ingredient
from sacred.run import Run
from sacred.serializer import flatten

from ingredients import plots as plots_ingredient

ingredient = Ingredient('history', ingredients=[plots_ingredient.ingredient])


@ingredient.capture
def load(name: str, path: str, _log: Logger):
    _log.info(f'Load train history [{name}]')
    with open(os.path.join(path, f'{name}.json'), 'r', encoding='utf8') as f:
        return json.loads(f.read())


@ingredient.capture
def save(name: str, history: dict, path: str, _log: Logger, _run: Run):
    _log.info(f'Save train history [{name}]')
    with open(os.path.join(path, f'{name}.json'), 'w', encoding='utf8') as f:
        f.write(json.dumps(flatten(history), ensure_ascii=False, indent=4))
        f.write('\n')


@ingredient.command
def plot(name: str, path: str, _log: Logger):
    _log.info(f'Plot train history [{name}].')

    history = load(name, path)
    x1_data: dict = {'ylabel': 'loss', 'xlabel': 'epoch', 'lines': []}
    x2_data: dict = {'ylabel': 'acc', 'lines': []}
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

    plots_ingredient.lines(name=name, x1_data=x1_data, x2_data=x2_data,
                           path=path)
