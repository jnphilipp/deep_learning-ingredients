# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os

from ingredients.data import ingredients


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, names=[]):
    print('Loading h5py [dataset=%s - which_set=%s]...' % (dataset, which_set))

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


@ingredients.capture
def len(DATASETS_DIR, dataset, which_set, name):
    with h5py.File(os.path.join(DATASETS_DIR, dataset, which_set), 'r') as f:
        ds = f[name]
        return ds.shape[0]


@ingredients.capture
def shape(DATASETS_DIR, dataset, which_set, name):
    with h5py.File(os.path.join(DATASETS_DIR, dataset, which_set), 'r') as f:
        ds = f[name]
        return ds.shape
