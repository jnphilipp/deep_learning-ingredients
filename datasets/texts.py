# -*- coding: utf-8 -*-

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


ingredient = Ingredient('datasets.texts', ingredients=[paths.ingredient])


class TextSequence(Sequence):
    def __init__(
        self,
        X: Dict[str, Tuple[int, List[List[int]]]],
        Y: Dict[str, Tuple[int, List[List[int]]]],
        rnd: np.random.RandomState,
        ids: Optional[List[str]] = None,
        batch_size: int = 10,
        sample_weights: bool = False,
        mode: str = 'separate',
        dtype: type = np.uint,
    ):
        assert mode in ['separate', 'concat']

        self.X = X
        self.Y = Y
        self.ids = ids
        self.size = len(X[list(X.keys())[0]][1])
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.mode = mode
        self.rnd = rnd
        self.dtype = dtype
        self.on_epoch_end()

    def on_epoch_end(self):
        self.index_array = self.rnd.permutation(self.size)

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]],
    ]:
        if self.size >= (idx * self.batch_size) + self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = self.size - (idx * self.batch_size)
        start_idx = idx * self.batch_size
        end_idx = start_idx + current_batch_size

        if self.mode == 'separate':
            bx = {}
            for k in self.X.keys():
                bx[k] = np.zeros((current_batch_size, self.X[k][0]), dtype=self.dtype)
            by = {}
            for k in self.Y.keys():
                by[k] = np.zeros((current_batch_size, self.Y[k][0]), dtype=self.dtype)

            if self.sample_weights:
                bw = {}
                for k in self.X.keys():
                    bw[k] = np.zeros((current_batch_size, self.X[k][0]), dtype=np.uint)

            for i, j in enumerate(range(start_idx, end_idx)):
                for k in self.X.keys():
                    bx[k][i, 0 : len(self.X[k][1][self.index_array[j]])] = self.X[k][1][
                        self.index_array[j]
                    ]
                for k in self.Y.keys():
                    by[k][i, 0 : len(self.Y[k][1][self.index_array[j]])] = self.Y[k][1][
                        self.index_array[j]
                    ]
                if self.sample_weights:
                    for k in self.X.keys():
                        bw[k][i, 0 : len(self.X[k][1][self.index_array[j]])] = 1

            if self.sample_weights:
                return bx, by, bw
            else:
                return bx, by
        elif self.mode == "concat":
            x = np.zeros(
                (current_batch_size, sum([v[0] for v in self.X.values()])),
                dtype=self.dtype,
            )
            y = np.zeros(
                (current_batch_size, sum([v[0] for v in self.Y.values()])),
                dtype=self.dtype,
            )
            if self.sample_weights:
                w = np.zeros(
                    (current_batch_size, sum([v[0] for v in self.X.values()])),
                    dtype=np.uint,
                )
            for i, j in enumerate(range(start_idx, end_idx)):
                length = sum([len(v[1][self.index_array[j]]) for v in self.X.values()])
                y_length = sum(
                    [len(v[1][self.index_array[j]]) for v in self.Y.values()]
                )
                x[i, 0:length] = list(
                    chain(v[1][self.index_array[j]] for v in self.X.values())
                )
                y[i, 0:y_length] = list(
                    chain(v[1][self.index_array[j]] for v in self.Y.values())
                )
                if self.sample_weights:
                    w[i, 0:length] = 1
            if self.sample_weights:
                return x, y, w
            else:
                return x, y
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
    dtype: type,
    _log: Logger,
    _rnd: np.random.RandomState,
    validation_split: Optional[float] = None,
    vocab_path: Optional[str] = None,
) -> Union[TextSequence, Tuple[TextSequence, TextSequence]]:
    assert mode in ['separate', 'concat']
    assert validation_split is None or (
        validation_split >= 0.0 and validation_split <= 1.0
    )

    vocab = None
    if vocab_path:
        vocab = json.vocab(os.path.join('{datasets_dir}', dataset, vocab_path))

    train_csv = os.path.join('{datasets_dir}', dataset, 'train.csv')
    tids, tx, ty = csv.load(
        train_csv,
        x_fieldnames,
        y_fieldnames,
        vocab=vocab,
        x_append_one=x_append_one,
        y_append_one=y_append_one,
        dtype=dtype,
    )
    train_x = {}
    for k in tx.keys():
        if tx[k]:
            max_len = max([len(i) for i in tx[k]])
            train_x[k if k.startswith('input_') else f'input_{k}'] = (max_len, tx[k])
    train_y = {}
    for k in ty.keys():
        if ty[k]:
            max_len = max([len(i) for i in ty[k]])
            train_y[k if k.startswith('output_') else f'output_{k}'] = (max_len, ty[k])

    val_csv = os.path.join('{datasets_dir}', dataset, 'val.csv')
    if os.path.exists(val_csv):
        vids, vx, vy = csv.load(
            val_csv,
            x_fieldnames,
            y_fieldnames,
            vocab=vocab,
            x_append_one=x_append_one,
            y_append_one=y_append_one,
            dtype=dtype,
        )
        val_x = {}
        for k in tx.keys():
            max_len = max(train_x[k][0], max([len(i) for i in tx[k]]))
            train_x[k] = (max_len, train_x[k][1])
            val_x[k if k.startswith('input_') else f'input_{k}'] = (max_len, vx[k])
        val_y = {}
        for k in ty.keys():
            max_len = max(train_y[k][0], max([len(i) for i in ty[k]]))
            train_y[k] = (max_len, train_y[k][1])
            val_y[k if k.startswith('output_') else f'output_{k}'] = (max_len, vy[k])

        for k in train_x.keys():
            _log.info(f"X[{k}] length: {train_x[k][0]}")
        for k in train_y.keys():
            _log.info(f"Y[{k}] length: {train_y[k][0]}")

        _log.info(
            f"Train on {len(train_x[list(train_x.keys())[0]][1])} samples and "
            + f"validating on {len(val_x[list(val_x.keys())[0]][1])} samples."
        )

        return (
            TextSequence(
                train_x, train_y, tids, batch_size, sample_weights, mode, dtype
            ),
            TextSequence(val_x, val_y, vids, batch_size, sample_weights, mode, dtype),
        )
    else:
        for k in train_x.keys():
            _log.info(f"X[{k}] length: {train_x[k][0]}")
        for k in train_y.keys():
            _log.info(f"Y[{k}] length: {train_y[k][0]}")

        if validation_split:
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
                train_x, train_y, _rnd, tids, batch_size, sample_weights, mode, dtype
            ), TextSequence(
                val_x, val_y, _rnd, vids, batch_size, sample_weights, mode, dtype
            )
        else:
            _log.info(f"Run on {len(train_x[list(train_x.keys())[0]][1])} samples.")
            return TextSequence(
                train_x, train_y, _rnd, tids, batch_size, sample_weights, mode, dtype
            )
