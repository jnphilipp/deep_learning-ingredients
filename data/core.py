# -*- coding: utf-8 -*-

import numpy as np

from ingredients.data import ingredients


@ingredients.capture
def split(X, y, validation_split):
    print('Making validation split [samples=%s - validation_split=%s]...' %
          (len(X), validation_split))

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for i in range(len(y[0])):
        index_array = np.where(y[:, i] == 1)
        split_at = int(len(index_array[0]) * (1. - validation_split))
        for i in index_array[0][:split_at]:
            X_train.append(X[i])
            y_train.append(y[i])
        for i in index_array[0][split_at:]:
            X_val.append(X[i])
            y_val.append(y[i])

    if len(set([x.shape for x in X_train])) <= 1:
        X_train = np.asarray(X_train)
    if len(set([x.shape for x in X_val])) <= 1:
        X_val = np.asarray(X_val)
    if len(set([x.shape for x in y_train])) <= 1:
        y_train = np.asarray(y_train)
    if len(set([x.shape for x in y_val])) <= 1:
        y_val = np.asarray(y_val)

    return X_train, y_train, X_val, y_val
