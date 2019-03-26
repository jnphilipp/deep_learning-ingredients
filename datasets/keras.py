# -*- coding: utf-8 -*-

from keras import backend as K, datasets
from keras.utils import to_categorical

from . import ingredient


@ingredient.capture
def mnist(data_format=None, _log=None):
    _log.info('Loading MNIST.')

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    nb_classes = 10
    rows, cols = 28, 28

    (X, y), (X_val, y_val) = datasets.mnist.load_data()
    y = to_categorical(y, nb_classes)
    y_val = to_categorical(y_val, nb_classes)

    if data_format == 'channels_first':
        X = X.reshape(X.shape[0], 1, rows, cols)
        X_val = X_val.reshape(X_val.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    elif data_format == 'channels_last':
        X = X.reshape(X.shape[0], rows, cols, 1)
        X_val = X_val.reshape(X_val.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    X = X.astype('float32')
    X_val = X_val.astype('float32')

    _log.info(f'{X.shape[0]} train samples.')
    _log.info(f'{X_val.shape[0]} validation samples.')

    return X, y, X_val, y_val


@ingredient.capture
def cifar10(_log):
    _log.info('Loading CIFAR10.')

    (X, y), (X_val, y_val) = datasets.cifar10.load_data()
    y = to_categorical(y, 10)
    y_val = to_categorical(y_val, 10)

    _log.info(f'{X.shape[0]} train samples.')
    _log.info(f'{X_val.shape[0]} validation samples.')

    return X, y, X_val, y_val
