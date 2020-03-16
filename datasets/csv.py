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

import numpy as np

from csv import DictReader
from libdlutils import utils
from logging import Logger
from sacred import Ingredient
from typing import Dict, List, Optional, Tuple

from .. import paths


ingredient = Ingredient('datasets.csv', ingredients=[paths.ingredient])


@ingredient.capture
def load(path: str, fieldnames: List[Tuple[str, str, str]], paths: Dict,
         _log: Logger, vocab: Optional[utils.Vocab] = None,
         append_one: bool = False, dtype: type = np.uint) -> \
        Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    def transform(field: str, vocab: Optional[utils.Vocab] = None) -> \
            np.ndarray:
        return np.array([(vocab.get(i) if vocab else i)
                         for i in field.split(',') if i] +
                        ([1] if append_one else []), dtype=dtype)

    _log.info(f'Load {path.format(datasets_dir=paths["datasets_dir"])}.')
    x: Dict[str, List[np.ndarray]] = {field[0]: [] for field in fieldnames}
    y: Dict[str, List[np.ndarray]] = {field[0]: [] for field in fieldnames}
    with open(path.format(datasets_dir=paths["datasets_dir"]), 'r',
              encoding='utf8') as f:
        reader = DictReader(f, dialect='unix')
        for row in reader:
            for field in fieldnames:
                x[field[0]].append(transform(row[field[1]], vocab))
                y[field[0]].append(transform(row[field[2]]))

    return x, y
