# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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
from logging import Logger
from sacred import Ingredient
from typing import Dict, List, Optional, Tuple, Union

from .. import paths
from ..vocab import Vocab


ingredient = Ingredient("datasets.csv", ingredients=[paths.ingredient])


@ingredient.capture
def load(
    path: str,
    x_fieldnames: Union[str, List[str]],
    y_fieldnames: Union[str, List[str]],
    paths: Dict,
    _log: Logger,
    id_fieldname: Optional[str] = "id",
    vocab: Optional[Vocab] = None,
    x_append_one: bool = True,
    y_append_one: bool = False,
    dtype: type = np.uint,
) -> Tuple[List[str], Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    def transform(
        field: str, append_one: bool, vocab: Optional[Vocab] = None
    ) -> np.ndarray:
        if ";" in field and "," in field:
            return np.array(
                [
                    (vocab.get(j) if vocab else j)
                    for i in field.split(";")
                    for j in i.split(",")
                    if j
                ]
                + ([1] if append_one else []),
                dtype=dtype,
            )
        else:
            return np.array(
                [(vocab.get(i) if vocab else i) for i in field.split(",") if i]
                + ([1] if append_one else []),
                dtype=dtype,
            )

    if isinstance(x_fieldnames, str):
        x_fieldnames = [x_fieldnames]
    if isinstance(y_fieldnames, str):
        y_fieldnames = [y_fieldnames]

    _log.info(f"Load {path.format(datasets_dir=paths['datasets_dir'])}.")
    ids = []
    x: Dict[str, List[np.ndarray]] = {k: [] for k in x_fieldnames}
    y: Dict[str, List[np.ndarray]] = {k: [] for k in y_fieldnames}
    with open(
        path.format(datasets_dir=paths["datasets_dir"]), "r", encoding="utf8"
    ) as f:
        reader = DictReader(f, dialect="unix")
        for row in reader:
            if id_fieldname and id_fieldname in row:
                ids.append(row[id_fieldname])
            for field in x_fieldnames:
                if field in row:
                    x[field].append(transform(row[field], x_append_one, vocab))
            for field in y_fieldnames:
                if field in row:
                    y[field].append(transform(row[field], y_append_one, vocab))

    return ids, x, y
