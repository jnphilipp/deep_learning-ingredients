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
"""datasets.texts ingredient."""

import math
import numpy as np
import os

from logging import Logger
from itertools import chain
from sacred import Ingredient
from tensorflow.keras.utils import Sequence
from typing import Dict, List, Optional, Tuple, Union

from . import csv, json
from .. import paths


ingredient = Ingredient("datasets.texts", ingredients=[paths.ingredient])


class TextSequence(Sequence):
    """Text sequence for tf.keras."""

    def __init__(
        self,
        x: Dict[str, Tuple[int, np.ndarray]],
        y: Dict[str, Tuple[int, np.ndarray]],
        rnd: np.random.RandomState,
        ids: Optional[List[str]] = None,
        batch_size: int = 10,
        sample_weights: bool = False,
        mode: str = "separate",
    ):
        """Create a new text sequence."""
        assert mode in ["separate", "concat"]

        self.x = x
        self.y = y
        self.ids = ids
        self.size = len(self.x[list(self.x.keys())[0]][1])
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.mode = mode
        self.rnd = rnd

        if self.mode == "separate":
            self.dtype = self.x[list(self.x.keys())[0]][1].dtype

        self.on_epoch_end()

    def on_epoch_end(self):
        """Shuffle data on epoch end."""
        self.index_array = self.rnd.permutation(self.size)

    def __len__(self) -> int:
        """Steps per epoch."""
        return math.ceil(self.size / self.batch_size)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]],
    ]:
        """Get batch at index."""
        if self.size >= (idx * self.batch_size) + self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = self.size - (idx * self.batch_size)
        start_idx = idx * self.batch_size
        end_idx = start_idx + current_batch_size

        bx: Dict[str, np.ndarray] = {}
        by: Dict[str, np.ndarray] = {}
        bw: Dict[str, np.ndarray] = {}
        if self.mode == "separate":
            for k in self.x.keys():
                bx[k] = np.zeros(
                    (current_batch_size, self.x[k][0]), dtype=self.self.x[k][1].dtype
                )
            for k in self.y.keys():
                by[k] = np.zeros(
                    (current_batch_size, self.y[k][0]), dtype=self.self.x[k][1].dtype
                )

            if self.sample_weights:
                for k in self.x.keys():
                    bw[k] = np.zeros((current_batch_size, self.x[k][0]), dtype=np.uint)

            for i, j in enumerate(range(start_idx, end_idx)):
                for k in self.x.keys():
                    bx[k][i, 0 : len(self.x[k][1][self.index_array[j]])] = self.x[k][1][
                        self.index_array[j]
                    ]
                for k in self.y.keys():
                    by[k][i, 0 : len(self.y[k][1][self.index_array[j]])] = self.y[k][1][
                        self.index_array[j]
                    ]
                if self.sample_weights:
                    for k in self.x.keys():
                        bw[k][i, 0 : len(self.x[k][1][self.index_array[j]])] = 1

            if self.sample_weights:
                return bx, by, bw
            else:
                return bx, by
        elif self.mode == "concat":
            bx["concat"] = np.zeros(
                (current_batch_size, sum([v[0] for v in self.x.values()])),
                dtype=self.dtype,
            )
            by["concat"] = np.zeros(
                (current_batch_size, sum([v[0] for v in self.y.values()])),
                dtype=self.dtype,
            )
            if self.sample_weights:
                bw["concat"] = np.zeros(
                    (current_batch_size, sum([v[0] for v in self.x.values()])),
                    dtype=np.uint,
                )
            for i, j in enumerate(range(start_idx, end_idx)):
                length = sum([len(v[1][self.index_array[j]]) for v in self.x.values()])
                y_length = sum(
                    [len(v[1][self.index_array[j]]) for v in self.y.values()]
                )
                bx["concat"][i, 0:length] = list(
                    chain(v[1][self.index_array[j]] for v in self.x.values())
                )
                by["concat"][i, 0:y_length] = list(
                    chain(v[1][self.index_array[j]] for v in self.y.values())
                )
                if self.sample_weights:
                    bw["concat"][i, 0:length] = 1
            if self.sample_weights:
                return bx["concat"], by["concat"], bw["concat"]
            else:
                return bx["concat"], by["concat"]
        else:
            return {}, {}


@ingredient.capture
def get(
    dataset: str,
    batch_size: int,
    mode: str,
    x_fieldnames: Union[str, List[str]],
    y_fieldnames: Union[str, List[str]],
    x_append_one: bool,
    y_append_one: bool,
    sample_weights: bool,
    dtype: Union[type, Dict[str, type]],
    _log: Logger,
    _rnd: np.random.RandomState,
    validation_split: Optional[float] = None,
    vocab_path: Optional[str] = None,
    vocab_fieldnames: List[str] = [],
    single_sequence: bool = False,
) -> Union[TextSequence, Tuple[TextSequence, TextSequence]]:
    """Get text sequenc(s) from csv data."""
    assert mode in ["separate", "concat"]
    assert validation_split is None or (
        validation_split >= 0.0 and validation_split <= 1.0
    )

    vocab = None
    if vocab_path:
        vocab = json.vocab(paths.join("{datasets_dir}", dataset, vocab_path))

    if os.path.isfile(paths.join("{datasets_dir}", dataset)):
        train_csv = paths.join("{datasets_dir}", dataset)
    else:
        train_csv = paths.join("{datasets_dir}", dataset, "train.csv")
    tids, tx, ty = csv.load(
        train_csv,
        x_fieldnames,
        y_fieldnames,
        vocab=vocab,
        vocab_fieldnames=vocab_fieldnames,
        x_append_one=x_append_one,
        y_append_one=y_append_one,
        dtype=dtype,
    )
    train_x = {}
    for k in tx.keys():
        if tx[k]:
            max_len = max([len(i) for i in tx[k]])
            train_x[k if k.startswith("input_") else f"input_{k}"] = (max_len, tx[k])
    train_y = {}
    for k in ty.keys():
        if ty[k]:
            max_len = max([len(i) for i in ty[k]])
            train_y[k if k.startswith("output_") else f"output_{k}"] = (max_len, ty[k])

    val_csv = paths.join("{datasets_dir}", dataset, "val.csv")
    if os.path.exists(val_csv):
        vids, vx, vy = csv.load(
            val_csv,
            x_fieldnames,
            y_fieldnames,
            vocab=vocab,
            vocab_fieldnames=vocab_fieldnames,
            x_append_one=x_append_one,
            y_append_one=y_append_one,
            dtype=dtype,
        )
        val_x = {}
        for k in tx.keys():
            xk = k if k.startswith("input_") else f"input_{k}"
            max_len = max(train_x[xk][0], max([len(i) for i in vx[k]]))
            train_x[xk] = (max_len, train_x[xk][1])
            val_x[xk] = (max_len, vx[k])
        val_y = {}
        for k in ty.keys():
            yk = k if k.startswith("output_") else f"output_{k}"
            max_len = max(train_y[yk][0], max([len(i) for i in vy[k]]))
            train_y[yk] = (max_len, train_y[yk][1])
            val_y[yk] = (max_len, vy[k])

        for k in train_x.keys():
            _log.info(f"X[{k}] length: {train_x[k][0]}")
        for k in train_y.keys():
            _log.info(f"Y[{k}] length: {train_y[k][0]}")

        if single_sequence:
            tids += vids
            for k in val_x.keys():
                train_x[k] = (
                    max(train_x[k][0], val_x[k][0]),
                    train_x[k][1] + val_x[k][1],
                )
            for k in val_y.keys():
                train_y[k] = (
                    max(train_y[k][0], val_y[k][0]),
                    train_y[k][1] + val_y[k][1],
                )

            _log.info(f"Train on {len(train_x[list(train_x.keys())[0]][1])} samples.")

            return TextSequence(
                train_x,
                train_y,
                _rnd,
                tids,
                batch_size,
                sample_weights,
                mode,
            )
        else:
            _log.info(
                f"Train on {len(train_x[list(train_x.keys())[0]][1])} samples and "
                + f"validating on {len(val_x[list(val_x.keys())[0]][1])} samples."
            )

            return (
                TextSequence(
                    train_x,
                    train_y,
                    _rnd,
                    tids,
                    batch_size,
                    sample_weights,
                    mode,
                ),
                TextSequence(
                    val_x,
                    val_y,
                    _rnd,
                    vids,
                    batch_size,
                    sample_weights,
                    mode,
                ),
            )
    else:
        for k in train_x.keys():
            _log.info(f"X[{k}] length: {train_x[k][0]}")
        for k in train_y.keys():
            _log.info(f"Y[{k}] length: {train_y[k][0]}")

        if validation_split and not single_sequence:
            length = len(train_x[list(train_x.keys())[0]][1])
            per = _rnd.permutation(length)
            idx = int(length * (1.0 - validation_split))

            vids = []
            for i in sorted(per[idx:], reverse=True):
                vids.append(tids[i])
                del tids[i]

            _log.info(
                f"Making validation split, train on {idx} samples and "
                + f"validating on {length - idx} samples."
            )

            val_x = {}
            for k in train_x.keys():
                val_x[k] = (train_x[k][0], [train_x[k][1][i] for i in per[idx:]])
                train_x[k] = (train_x[k][0], [train_x[k][1][i] for i in per[0:idx]])
            val_y = {}
            for k in train_y.keys():
                val_y[k] = (train_y[k][0], [train_y[k][1][i] for i in per[idx:]])
                train_y[k] = (train_y[k][0], [train_y[k][1][i] for i in per[0:idx]])

            return TextSequence(
                train_x, train_y, _rnd, tids, batch_size, sample_weights, mode
            ), TextSequence(val_x, val_y, _rnd, vids, batch_size, sample_weights, mode)
        else:
            _log.info(f"Run on {len(train_x[list(train_x.keys())[0]][1])} samples.")
            return TextSequence(
                train_x, train_y, _rnd, tids, batch_size, sample_weights, mode
            )
