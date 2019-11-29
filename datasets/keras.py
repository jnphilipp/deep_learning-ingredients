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
