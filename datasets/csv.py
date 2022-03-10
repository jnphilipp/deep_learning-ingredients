# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
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
"""datasets.csv ingredient."""

import gzip
import numpy as np

from csv import DictReader
from logging import Logger
from sacred import Ingredient
from tensorflow.keras.utils import to_categorical
from typing import Dict, List, Optional, Tuple, Union

from .. import paths
from ..vocab import Vocab


ingredient = Ingredient("datasets.csv", ingredients=[paths.ingredient])


@ingredient.capture
def load(
    path: str,
    x_fieldnames: Union[str, List[str]],
    y_fieldnames: Union[str, List[str]],
    _log: Logger,
    id_fieldname: Optional[str] = "id",
    vocab: Optional[Vocab] = None,
    vocab_fieldnames: List[str] = [],
    to_categorical_fieldnames: List[str] = [],
    x_append_one: Union[bool, List[str]] = True,
    y_append_one: Union[bool, List[str]] = False,
    dtype: Union[type, Dict[str, type]] = np.uint,
) -> Tuple[List[str], Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """Load data from csv."""

    def transform(
        field: str,
        append_one: bool,
        vocab: Optional[Vocab] = None,
        dtype: type = np.uint,
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
        elif ";" not in field and "," in field:
            return np.array(
                [(vocab.get(i) if vocab else i) for i in field.split(",") if i]
                + ([1] if append_one else []),
                dtype=dtype,
            )
        elif ";" in field and "," not in field:
            return np.array(
                [(vocab.get(i) if vocab else i) for i in field.split(";") if i]
                + ([1] if append_one else []),
                dtype=dtype,
            )
        else:
            return np.array(
                [(vocab.get(field) if vocab else field)] + ([1] if append_one else []),
                dtype=dtype,
            )

    if isinstance(x_fieldnames, str):
        x_fieldnames = [x_fieldnames]
    if isinstance(y_fieldnames, str):
        y_fieldnames = [y_fieldnames]

    _log.info(f"Load {paths.join(path)}.")
    ids = []
    x: Dict[str, List[np.ndarray]] = {k: [] for k in x_fieldnames}
    y: Dict[str, List[np.ndarray]] = {k: [] for k in y_fieldnames}

    if path.endswith(".gz"):
        f = gzip.open(paths.join(path), "rt", encoding="utf8")
    else:
        f = open(paths.join(path), "rt", encoding="utf8")
    reader = DictReader(f, dialect="unix")
    for row in reader:
        if id_fieldname and id_fieldname in row:
            ids.append(row[id_fieldname])
        try:
            for field in x_fieldnames:
                if field in row:
                    x[field].append(
                        transform(
                            row[field],
                            x_append_one
                            if isinstance(x_append_one, bool)
                            else (field in x_append_one),
                            vocab if field in vocab_fieldnames else None,
                            dtype[field] if isinstance(dtype, dict) else dtype,
                        )
                    )
            for field in y_fieldnames:
                if field in row:
                    y[field].append(
                        transform(
                            row[field],
                            y_append_one
                            if isinstance(y_append_one, bool)
                            else (field in y_append_one),
                            vocab if field in vocab_fieldnames else None,
                            dtype[field] if isinstance(dtype, dict) else dtype,
                        )
                    )
        except Exception as e:
            _log.error(
                f"Couldn't transform {field} field in row "
                + f"{ids[-1] if id_fieldname and id_fieldname in row else row}."
            )
            f.close()
            raise e
    f.close()

    for field in to_categorical_fieldnames:
        if field in x:
            x[field] = to_categorical(x[field])
        if field in y:
            y[field] = to_categorical(y[field])

    return ids, x, y
