# -*- coding: utf-8 -*-

from keras import backend as K, datasets
from keras.utils import to_categorical

from . import ingredient


@ingredient.capture
def mnist(_log):
    _log.info('Loading MNIST.')

    nb_classes = 10
    rows, cols = 28, 28

    (X, y), (X_val, y_val) = datasets.mnist.load_data()
    y = to_categorical(y, nb_classes)
    y_val = to_categorical(y_val, nb_classes)

    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, rows, cols)
        X_val = X_val.reshape(X_val.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        X = X.reshape(X.shape[0], rows, cols, 1)
        X_val = X_val.reshape(X_val.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    X = X.astype('float32')
    X_val = X_val.astype('float32')
    validation_data = (X_val, y_val)

    _log.info('%s train samples.' % X.shape[0])
    _log.info('%s validation samples.' % X_val.shape[0])

    return X, y, validation_data, input_shape, nb_classes
