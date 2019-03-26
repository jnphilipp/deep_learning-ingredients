# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sacred import Ingredient

from . import datasets

ingredient = Ingredient('sequences', ingredients=[datasets.ingredient])


@ingredient.config
def config():
    terminateonnan = True


@ingredient.capture
def get(sequence_type, batch_size, train_samples=None, validation_samples=None,
        train_image_datagen_args={}, validation_image_datagen_args={},
        _log=None, _run=None):
    assert sequence_type in {'cifar10', 'landkarten', 'mnist'}

    if sequence_type == 'cifar10':
        X, y, X_val, y_val = datasets.keras.cifar10()

        train_image_datagen = ImageDataGenerator(**train_image_datagen_args)
        train_image_datagen.fit(X)
        train_generator = train_image_datagen.flow(X, y, batch_size=batch_size)

        val_image_datagen = ImageDataGenerator(**validation_image_datagen_args)
        val_image_datagen.fit(X_val)
        validation_generator = val_image_datagen.flow(X, y,
                                                      batch_size=batch_size)
    elif sequence_type == 'landkarten':
        assert train_samples is not None and validation_samples is not None
        from_directory_args = {'dataset': 'landkarten', 'which_set': 'teile'}
        train_generator = LandkartenSequence(
            train_samples, batch_size, from_directory_args=from_directory_args,
            image_datagen_args=train_image_datagen_args)

        validation_generator = LandkartenSequence(
            train_samples, batch_size, from_directory_args=from_directory_args,
            image_datagen_args=validation_image_datagen_args)
    elif sequence_type == 'mnist':
        X, y, X_val, y_val = datasets.keras.mnist()

        train_image_datagen = ImageDataGenerator(**train_image_datagen_args)
        train_image_datagen.fit(X)
        train_generator = train_image_datagen.flow(X, y, batch_size=batch_size)

        val_image_datagen = ImageDataGenerator(**validation_image_datagen_args)
        val_image_datagen.fit(X_val)
        validation_generator = val_image_datagen.flow(X, y,
                                                      batch_size=batch_size)
    return train_generator, validation_generator


class LandkartenSequence(Sequence):
    def __init__(self, samples, batch_size, shape=None, from_directory_args={},
                 image_datagen_args={}, data_format=None):
        self.samples = samples
        self.batch_size = batch_size
        self.data_format = data_format
        if self.data_format is None:
            self.data_format = K.image_data_format()
        if self.data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format:', self.data_format)
        self.image_data_generator = ImageDataGenerator(**image_datagen_args)
        self.shape = shape
        self.X, self.y = datasets.images.from_directory(**from_directory_args)
        self.steps = np.ceil(self.samples / self.batch_size)
        self.on_epoch_end()

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, shape):
        self.__shape = shape
        if self.shape is not None:
            if self.data_format == 'channels_first':
                self.r = self.shape[1]
                self.c = self.shape[2]
            elif self.data_format == 'channels_last':
                self.r = self.shape[0]
                self.c = self.shape[1]

    def on_epoch_end(self):
        self.index_array = []
        for i in range(int(np.floor(self.samples / float(len(self.X))))):
            perm = np.random.permutation(len(self.X))
            self.index_array += list(perm)

            nb_rest = self.samples - len(self.index_array)
            if nb_rest > 0:
                perm = np.random.permutation(len(self.X))
                self.index_array += list(perm)[:nb_rest]

    def __getitem__(self, index):
        assert self.shape is not None

        if self.samples >= (index * self.batch_size) + self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = self.samples - (index * self.batch_size)

        bX = np.zeros((current_batch_size,) + tuple(self.shape))
        if type(self.y) == dict:
            by = {k: np.zeros((current_batch_size,) + v.shape[1:])
                  for k, v in self.y.items()}
        else:
            by = np.zeros((current_batch_size,) + self.y.shape[1:])

        for i, j in enumerate(self.index_array[index * self.batch_size:
                              index * self.batch_size + current_batch_size]):
            x = self.X[j]
            if self.data_format == 'channels_first':
                rows = x.shape[1]
                cols = x.shape[2]
            elif self.data_format == 'channels_last':
                rows = x.shape[0]
                cols = x.shape[1]

            tlr = np.random.randint(0, rows - self.r) if rows > self.r else 0
            tlc = np.random.randint(0, cols - self.c) if cols > self.c else 0

            if self.data_format == 'channels_first':
                x = x[:, tlr:tlr + self.r, tlc:tlc + self.c]
            elif self.data_format == 'channels_last':
                x = x[tlr:tlr + self.r, tlc:tlc + self.c, :]

            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.standardize(
                self.image_data_generator.apply_transform(x, params))

            bX[i] = x
            if type(self.y) == dict:
                for k in self.y.keys():
                    by[k][i] = self.y[k][j]
            else:
                by[i] = self.y[j]

        return bX, by

    def __len__(self):
        return int(np.ceil(self.samples / float(self.batch_size)))
