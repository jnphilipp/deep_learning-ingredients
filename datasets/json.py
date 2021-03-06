# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020
#               J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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

from logging import Logger
from sacred import Ingredient
from typing import Dict, List, Optional, Tuple

from .. import paths
from ..vocab import Vocab


ingredient = Ingredient('datasets.json', ingredients=[paths.ingredient])


@ingredient.capture
def load(path: str, paths: Dict, _log: Logger) -> Dict:
    _log.info('Loading json from ' +
              f'{path.format(datasets_dir=paths["datasets_dir"])}.')
    with open(path.format(datasets_dir=paths["datasets_dir"]), 'r',
              encoding='utf-8') as f:
        return json.loads(f.read())


@ingredient.capture
def vocab(path: str, paths: Dict, _log: Logger) -> Vocab:
    _log.info('Load vocab from ' +
              f'{path.format(datasets_dir=paths["datasets_dir"])}.')
    return Vocab.load(path.format(datasets_dir=paths["datasets_dir"]))
