# -*- coding: utf-8 -*-

import cairocffi as cairo
import string
import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, random_rotation
from keras.utils import Sequence, np_utils
from sacred import Ingredient

from . import datasets

ingredient = Ingredient('sequences', ingredients=[datasets.ingredient])


@ingredient.config
def config():
    terminateonnan = True


@ingredient.capture
def get(sequence_type, batch_size, train_samples=None, validation_samples=None,
        train_image_datagen_args={}, validation_image_datagen_args={},
        artificial_split=0., artificial_config={}, _log=None, _run=None):
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
        X, y = datasets.images.from_directory(dataset='landkarten',
                                              which_set='karten',
                                              mask_names=list('abcdefg'))

        train_generator = LandkartenSequence(
            X, y, train_samples, batch_size, artificial_split=artificial_split,
            image_datagen_args=train_image_datagen_args,
            artificial_config=artificial_config)

        validation_generator = LandkartenSequence(
            X, y, validation_samples, batch_size,
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
    def __init__(self, X, y, samples, batch_size, shape=None,
                 image_datagen_args={}, artificial_split=0.,
                 artificial_config={}, data_format=None):
        self.X = X
        self.y = y
        self.samples = samples
        self.batch_size = batch_size
        self.data_format = data_format
        if self.data_format is None:
            self.data_format = K.image_data_format()
        if self.data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format:', self.data_format)
        self.image_datagen = ImageDataGenerator(**image_datagen_args)
        self.shape = shape
        self.steps = np.ceil(self.samples / self.batch_size)

        if artificial_split is None or artificial_split < 0.:
            artificial_split = 0.
        elif artificial_split > 1.:
            artificial_split = 1.
        self.artificial_split = artificial_split

        if artificial_config is not None and \
                self.artificial_split > 0.:
            if 'fonts' in artificial_config:
                self.fonts = artificial_config['fonts']
            else:
                raise ValueError('Missing config value "fonts".')
            if 'font_rg' in artificial_config:
                self.font_rg = artificial_config['font_rg']
            else:
                raise ValueError('Missing config value "font_rg".')

            if 'font_size' in artificial_config:
                if type(artificial_config['font_size']) == str:
                    self.font_size = eval(artificial_config['font_size'])
                else:
                    self.font_size = artificial_config['font_size']
            else:
                raise ValueError('Missing config value "font_size".')

            if 'font_rgb' in artificial_config:
                if type(artificial_config['font_rgb']) == str:
                    self.font_rgb = eval(artificial_config['font_rgb'])
                else:
                    self.font_rgb = artificial_config['font_rgb']
            else:
                raise ValueError('Missing config value "font_rgb".')
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

    def choose(self, j):
        if self.data_format == 'channels_first':
            rows = self.X[j].shape[1]
            cols = self.X[j].shape[2]
            shape = (3, self.r, self.c)
        elif self.data_format == 'channels_last':
            rows = self.X[j].shape[0]
            cols = self.X[j].shape[1]
            shape = (self.r, self.c, 3)

        while True:
            tlr = np.random.randint(0, rows - self.r) if rows > self.r else 0
            tlc = np.random.randint(0, cols - self.c) if cols > self.c else 0

            text = np.zeros(shape)
            params = self.image_datagen.get_random_transform(text.shape)
            for k in self.y.keys():
                text += self.transform(self.extract(self.y[k][j], tlr, tlc),
                                       params)

            if len(text[np.where(text > 0.6)]) / text.size >= 0.02:
                return tlr, tlc, params

    def extract(self, x, tlr, tlc):
        if self.data_format == 'channels_first':
            return x[:, tlr:tlr + self.r, tlc:tlc + self.c].copy()
        elif self.data_format == 'channels_last':
            return x[tlr:tlr + self.r, tlc:tlc + self.c, :].copy()

    def generate(self, text, height, width):
        """Paints the string in a random location in a random font, with a
        slight random rotation
        """

        def rand_shift(max_shift):
            return np.random.normal(max_shift // 2, max(max_shift // 4, 0.001))

        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
        with cairo.Context(surface) as context:
            context.set_source_rgb(1., 1., 1.)
            context.paint()

            context.select_font_face(
                np.random.choice(self.fonts),
                np.random.choice([cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_SLANT_ITALIC,
                                  cairo.FONT_SLANT_OBLIQUE]),
                np.random.choice([cairo.FONT_WEIGHT_BOLD,
                                  cairo.FONT_WEIGHT_NORMAL])
            )

            if callable(self.font_size):
                context.set_font_size(self.font_size())
            else:
                context.set_font_size(self.font_size)
            box = context.text_extents(text)

            max_shift_x = (width // 3) - abs(box[0])
            top_left_x = min(max(rand_shift(max_shift_x), abs(box[0])), width)

            max_shift_y = height - abs(box[1])
            top_left_y = min(max(rand_shift(max_shift_y), abs(box[1])), height)

            context.move_to(top_left_x, top_left_y)
            if callable(self.font_rgb):
                rgb = self.font_rgb()
                context.set_source_rgb(rgb[0], rgb[1], rgb[2])
            else:
                context.set_source_rgb(self.font_rgb[0], self.font_rgb[1],
                                       self.font_rgb[2])
            context.show_text(text)

        buf = surface.get_data()
        x = np.frombuffer(buf, np.uint8)
        x.shape = (height, width, 4)
        x = x[..., [2, 1, 0]]
        if self.data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
            channel_axis = 0
            row_axis = 1
            col_axis = 2
        elif self.data_format == 'channels_last':
            channel_axis = 2
            row_axis = 0
            col_axis = 1

        rg = (-self.font_rg - self.font_rg) * np.random.random() + self.font_rg
        x = random_rotation(x.astype(np.float32), rg, row_axis=row_axis,
                            col_axis=col_axis, channel_axis=channel_axis)
        if self.data_format == 'channels_first':
            grayscale = np.dot(x[:3, ...], [0.299, 0.587, 0.114])
            grayscale = grayscale.reshape((1,) + grayscale.shape)

            x[0, ...][x[0, ...] == 255.] = 247.
            x[1, ...][x[1, ...] == 255.] = 226.
            x[2, ...][x[2, ...] == 255.] = 185.
        elif self.data_format == 'channels_last':
            grayscale = np.dot(x[..., :3], [0.299, 0.587, 0.114])
            grayscale = grayscale.reshape(grayscale.shape + (1,))

            x[..., 0][x[..., 0] == 255.] = 247.
            x[..., 1][x[..., 1] == 255.] = 226.
            x[..., 2][x[..., 2] == 255.] = 185.

        grayscale[grayscale < 225.] = 0.
        grayscale[grayscale >= 225.] = 255.
        mask = 255. - grayscale
        if len(mask[mask == 255.]) == 0:
            return self.generate(text, height, width)
        else:
            return x, mask

    def on_epoch_end(self):
        nb_maps = int(np.ceil(self.samples * (1 - self.artificial_split)))

        self.sum_maps = 0
        self.index_array = np.random.permutation(len(self.X))
        self.index_array = np.repeat(self.index_array,
                                     int(np.floor(nb_maps / len(self.X))))
        if len(self.index_array) != nb_maps:
            self.index_array = np.append(self.index_array, np.array(
                [self.index_array[-1]] * (nb_maps - len(self.index_array))))

    def transform(self, x, params):
        return self.image_datagen.standardize(
            self.image_datagen.apply_transform(x, params))

    def __getitem__(self, index):
        assert self.shape is not None

        if self.samples >= (index * self.batch_size) + self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = self.samples - (index * self.batch_size)
        current_maps = int(np.ceil(current_batch_size *
                                   (1 - self.artificial_split)))
        current_artificial = int(np.floor(current_batch_size *
                                          self.artificial_split))

        bX = np.zeros((current_batch_size,) + tuple(self.shape))
        by = {'symbols': None, 'text': None}
        if self.data_format == 'channels_first':
            by['symbols'] = np.zeros((current_batch_size, len(self.y.keys()),
                                      self.r, self.c))
            by['text'] = np.zeros((current_batch_size, 1, self.r, self.c))
        elif self.data_format == 'channels_last':
            by['symbols'] = np.zeros((current_batch_size, self.r, self.c,
                                      len(self.y.keys())))
            by['text'] = np.zeros((current_batch_size, self.r, self.c, 1))

        self.sum_maps += current_maps
        for i, j in enumerate(self.index_array[self.sum_maps:
                                               self.sum_maps + current_maps]):
            tlr, tlc, params = self.choose(j)
            bX[i] = self.transform(self.extract(self.X[j], tlr, tlc), params)
            for a, k in enumerate(self.y.keys()):
                mask = self.transform(self.extract(self.y[k][j], tlr, tlc),
                                      params)
                if self.data_format == 'channels_first':
                    by['symbols'][i, a, ...] = mask[0, ...]
                elif self.data_format == 'channels_last':
                    by['symbols'][i, ..., a] = mask[..., 0]
                by['text'][i] += mask

        for i in range(current_maps, current_maps + current_artificial):
            text = ''.join(np.random.choice([
                ['a', 'A'],
                ['b', 'B'],
                ['c', 'C'],
                ['d', 'D'],
                ['e', 'E'],
                ['f', 'F'],
                ['g', 'G'],
            ][np.random.choice(3)], 2))

            while True:
                x, mask = self.generate(text, self.r, self.c)

                params = self.image_datagen.get_random_transform(x.shape)
                x = self.transform(x, params)
                mask = self.transform(mask, params)

                if len(mask[np.where(mask > 0.6)]) / mask.size >= 0.02:
                        break

            bX[i] = x
            for a, k in enumerate(self.y.keys()):
                if text[0].lower() == k:
                    by['symbols'][i, ..., a] = mask[..., 0]
                    by['text'][i] += mask

        permutation = np.random.permutation(len(bX))
        bX = bX[permutation]
        for k in by.keys():
            by[k] = by[k][permutation]
        return bX, by

    def __len__(self):
        return int(np.ceil(self.samples / float(self.batch_size)))
