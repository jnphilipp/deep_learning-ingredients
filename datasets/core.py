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

import math
import numpy as np

from . import ingredient


@ingredient.capture
def split(X, y=None, validation_split=0.25, _log=None, *args, **kwargs):
    _log.info('Making validation split [samples=%s: validation split=%s].' %
              (len(X), validation_split))

    if y is not None:
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        splits = {}
        for i, a in enumerate(args):
            splits['%s_train' % i] = []
            splits['%s_val' % i] = []
        for k in kwargs.keys():
            splits['%s_train' % k] = []
            splits['%s_val' % k] = []
        for i in range(len(y[0])):
            index_array = np.where(y[:, i] == 1)
            split_at = int(len(index_array[0]) * (1. - validation_split))
            for i in index_array[0][:split_at]:
                X_train.append(X[i])
                y_train.append(y[i])
            for i in index_array[0][split_at:]:
                X_val.append(X[i])
                y_val.append(y[i])
            for i, a in enumerate(args):
                for j in index_array[0][:split_at]:
                    splits['%s_train' % i].append(a[j])
                for j in index_array[0][split_at:]:
                    splits['%s_val' % i].append(a[j])
            for k in kwargs.keys():
                for i in index_array[0][:split_at]:
                    splits['%s_train' % k].append(kwargs[k][i])
                for i in index_array[0][split_at:]:
                    splits['%s_val' % k].append(kwargs[k][i])

        if len(set([x.shape for x in X_train])) <= 1:
            X_train = np.asarray(X_train)
        if len(set([x.shape for x in X_val])) <= 1:
            X_val = np.asarray(X_val)
        if len(set([x.shape for x in y_train])) <= 1:
            y_train = np.asarray(y_train)
        if len(set([x.shape for x in y_val])) <= 1:
            y_val = np.asarray(y_val)

        train = [X_train, y_train]
        val = [X_val, y_val]
        for i, a in enumerate(args):
            if len(set([x.shape for x in splits['%s_train' % i]])) <= 1:
                splits['%s_train' % i] = np.asarray(splits['%s_train' % i])
            if len(set([x.shape for x in splits['%s_val' % i]])) <= 1:
                splits['%s_val' % i] = np.asarray(splits['%s_val' % i])
            train.append(splits['%s_train' % i])
            val.append(splits['%s_val' % i])
        for k in kwargs.keys():
            if len(set([x.shape for x in splits['%s_train' % k]])) <= 1:
                splits['%s_train' % k] = np.asarray(splits['%s_train' % k])
            if len(set([x.shape for x in splits['%s_val' % k]])) <= 1:
                splits['%s_val' % k] = np.asarray(splits['%s_val' % k])
            train.append(splits['%s_train' % k])
            val.append(splits['%s_val' % k])

        _log.info('Train on %d samples, validate on %d samples.' %
                  (len(X_train), len(X_val)))
        return train + val
    else:
        validation_samples = math.floor(len(X) * validation_split)

        X_train = X[:validation_samples * -1]
        X_val = X[len(X) - validation_samples:]

        _log.info('Train on %d samples, validate on %d samples.' %
                  (len(X_train), len(X_val)))
        return X_train, X_val
