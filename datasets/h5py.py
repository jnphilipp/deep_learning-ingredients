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

import h5py
import numpy as np
import os

from . import ingredient


@ingredient.capture
def load(DATASETS_DIR, dataset, which_set, _log, names=[]):
    _log.info('Loading h5py [%s: %s].' % (dataset, which_set))

    with h5py.File(os.path.join(DATASETS_DIR, dataset, which_set), 'r') as f:
        if names:
            matrices = []
            for name in names:
                ds = f[name]
                m = np.zeros(ds.shape, dtype=ds.dtype)
                ds.read_direct(m)
                matrices.append(m)
            return matrices
        else:
            matrices = {}
            for k in f.keys():
                ds = f[k]
                m = np.zeros(ds.shape, dtype=ds.dtype)
                ds.read_direct(m)
                matrices[k] = m
            return matrices


@ingredient.capture
def len(DATASETS_DIR, dataset, which_set, name):
    with h5py.File(os.path.join(DATASETS_DIR, dataset, which_set), 'r') as f:
        ds = f[name]
        return ds.shape[0]


@ingredient.capture
def shape(DATASETS_DIR, dataset, which_set, name):
    with h5py.File(os.path.join(DATASETS_DIR, dataset, which_set), 'r') as f:
        ds = f[name]
        return ds.shape


@ingredient.capture
def save(path, name, matrices):
    assert type(matrices) == dict

    with h5py.File(path, 'w') as f:
        f.attrs['name'] = name
        for k, v in matrices.items():
            param_dset = f.create_dataset(k, v.shape, dtype=v.dtype)
            param_dset[:] = v
            f.flush()
