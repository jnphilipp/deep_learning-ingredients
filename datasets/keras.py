# -*- coding: utf-8 -*-

from logging import Logger
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

from . import ingredient


@ingredient.capture
def mnist(_log: Logger):
    _log.info('Loading MNIST.')

    nb_classes = 10
    rows, cols = 28, 28

    (x_train, y_train), (x_val, y_val) = datasets.mnist.load_data()
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)

    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_val = x_val.reshape(x_val.shape[0], rows, cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    _log.info(f'{x_train.shape[0]} train samples.')
    _log.info(f'{x_val.shape[0]} validation samples.')

    return x_train, y_train, x_val, y_val


@ingredient.capture
def cifar10(_log):
    _log.info('Loading CIFAR10.')

    (x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    _log.info(f'{x_train.shape[0]} train samples.')
    _log.info(f'{x_val.shape[0]} validation samples.')

    return x_train, y_train, x_val, y_val
