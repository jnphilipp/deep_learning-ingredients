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
        artificial_maps_split=0., artificial_maps_config={}, _log=None,
        _run=None):
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
        bg_maps_from_directory_args = {'dataset': 'landkarten',
                                       'which_set': 'teile/bg'}
        train_generator = LandkartenSequence(
            train_samples, batch_size, from_directory_args=from_directory_args,
            image_datagen_args=train_image_datagen_args,
            bg_maps_from_directory_args=bg_maps_from_directory_args,
            artificial_maps_split=artificial_maps_split,
            artificial_maps_config=artificial_maps_config)

        validation_generator = LandkartenSequence(
            validation_samples, batch_size,
            from_directory_args=from_directory_args,
            image_datagen_args=validation_image_datagen_args,
            bg_maps_from_directory_args=bg_maps_from_directory_args,
            artificial_maps_split=artificial_maps_split,
            artificial_maps_config=artificial_maps_config)
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
                 bg_maps_from_directory_args={}, image_datagen_args={},
                 artificial_maps_split=0., artificial_maps_config={},
                 data_format=None):
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

        if artificial_maps_split is None or artificial_maps_split < 0.:
            artificial_maps_split = 0.
        elif artificial_maps_split > 1.:
            artificial_maps_split = 1.
        self.artificial_maps_split = artificial_maps_split
        self.nb_maps = int(np.ceil(self.samples *
                                   (1 - self.artificial_maps_split)))
        self.nb_artificial_maps = int(np.floor(self.samples *
                                               self.artificial_maps_split))

        if artificial_maps_config is not None and \
                self.artificial_maps_split > 0.:
            if 'fonts' in artificial_maps_config:
                self.fonts = artificial_maps_config['fonts']
            else:
                raise ValueError('Missing config value "fonts".')
            if 'font_rg' in artificial_maps_config:
                self.font_rg = artificial_maps_config['font_rg']
            else:
                raise ValueError('Missing config value "font_rg".')

            if 'font_size' in artificial_maps_config:
                if type(artificial_maps_config['font_size']) == str:
                    self.font_size = eval(artificial_maps_config['font_size'])
                else:
                    self.font_size = artificial_maps_config['font_size']
            else:
                raise ValueError('Missing config value "font_size".')

            if 'font_rgb' in artificial_maps_config:
                if type(artificial_maps_config['font_rgb']) == str:
                    self.font_rgb = eval(artificial_maps_config['font_rgb'])
                else:
                    self.font_rgb = artificial_maps_config['font_rgb']
            else:
                raise ValueError('Missing config value "font_rgb".')

            if 'text_func' in artificial_maps_config:
                if type(artificial_maps_config['text_func']) == str:
                    self.text_func = eval(artificial_maps_config['text_func'])
                else:
                    self.text_func = artificial_maps_config['text_func']
            else:
                raise ValueError('Missing config value "text_func".')

            self.bg_maps = datasets.images.from_directory(
                **bg_maps_from_directory_args)

        self.sum_nb_maps = 0
        self.sum_nb_artificial_maps = 0
        self.on_epoch_end()

    @property
    def shape(self):
        return self.__shape

    @property
    def output_shapes(self):
        return self.__output_shapes

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

    @output_shapes.setter
    def output_shapes(self, output_shapes):
        self.__output_shapes = output_shapes

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
        a = np.frombuffer(buf, np.uint8)
        a.shape = (height, width, 4)
        if self.data_format == 'channels_first':
            a = a[0:3, :, :]
            a = a.transpose(2, 0, 1)
            channel_axis = 0
            row_axis = 1
            col_axis = 2
        elif self.data_format == 'channels_last':
            a = a[:, :, 0:3]
            channel_axis = 2
            row_axis = 0
            col_axis = 1

        rg = (-self.font_rg - self.font_rg) * np.random.random() + self.font_rg
        return random_rotation(a.astype(np.float32), rg, row_axis=row_axis,
                               col_axis=col_axis, channel_axis=channel_axis)

    def generate_mask(self, x, shape):
        if self.data_format == 'channels_first':
            grayscale = np.dot(x[:3, ...], [0.299, 0.587, 0.114])
        elif self.data_format == 'channels_last':
            grayscale = np.dot(x[..., :3], [0.299, 0.587, 0.114])
        grayscale[grayscale < 225.] = 0.
        grayscale[grayscale >= 225.] = 255.

        mask = np.zeros(shape)
        if self.data_format == 'channels_first':
            if shape[0] == 1:
                mask[0] = 255. - grayscale
            else:
                mask[0] = grayscale
                mask[1] = 255. - grayscale
        elif self.data_format == 'channels_last':
            if shape[-1] == 1:
                mask[:, :, 0] = 255. - grayscale
            else:
                mask[:, :, 0] = grayscale
                mask[:, :, 1] = 255. - grayscale
        return mask

    def on_epoch_end(self):
        self.index_array = []
        for i in range(int(np.floor(self.nb_maps / float(len(self.X))))):
            perm = np.random.permutation(len(self.X))
            self.index_array += list(perm)

            nb_rest = self.nb_maps - len(self.index_array)
            if nb_rest > 0:
                perm = np.random.permutation(len(self.X))
                self.index_array += list(perm)[:nb_rest]
        self.index_array = np.random.permutation(self.index_array)

    def __getitem__(self, index):
        assert self.shape is not None

        if self.samples >= (index * self.batch_size) + self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = self.samples - (index * self.batch_size)

        bX = np.zeros((current_batch_size,) + tuple(self.shape))
        by = {}
        for k, v in self.y.items():
            by[k] = np.zeros((current_batch_size,) +
                             tuple(self.output_shapes[k]))

        current_nb_maps = int(np.ceil(current_batch_size *
                                      (1 - self.artificial_maps_split)))
        current_nb_artificial_maps = int(np.floor(current_batch_size *
                                                  self.artificial_maps_split))

        self.sum_nb_maps += current_nb_maps
        self.sum_nb_artificial_maps += current_nb_artificial_maps

        for i, j in enumerate(self.index_array[self.sum_nb_maps:
                              self.sum_nb_maps + current_nb_maps]):
            x = self.X[j]
            if self.data_format == 'channels_first':
                rows = x.shape[1]
                cols = x.shape[2]
                mask_shape = (1, rows, cols)
            elif self.data_format == 'channels_last':
                rows = x.shape[0]
                cols = x.shape[1]
                mask_shape = (rows, cols, 1)

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
            for k in self.y.keys():
                if self.y[k][j].shape == mask_shape:
                    if self.data_format == 'channels_first':
                        mask = self.y[k][j][:, tlr:tlr + self.r,
                                            tlc:tlc + self.c]
                    elif self.data_format == 'channels_last':
                        mask = self.y[k][j][tlr:tlr + self.r,
                                            tlc:tlc + self.c, :]
                    by[k][i] = self.image_data_generator.standardize(
                        self.image_data_generator.apply_transform(mask,
                                                                  params))
                else:
                    by[k][i] = self.y[k][j]

        for i in range(current_nb_maps,
                       current_nb_maps + current_nb_artificial_maps):
            y = np.random.choice([0, 1])
            if y == 0:
                x = self.generate('', self.r, self.c)
            else:
                x = self.generate(self.text_func(), self.r, self.c)

            if 'mask' in self.y:
                mask = self.generate_mask(x, self.output_shapes['mask'])

            bg = self.bg_maps[np.random.choice(len(self.bg_maps))]
            if self.data_format == 'channels_first':
                rows = bg.shape[1]
                cols = bg.shape[2]
            elif self.data_format == 'channels_last':
                rows = bg.shape[0]
                cols = bg.shape[1]

            tlr = np.random.randint(0, rows - self.r) if rows > self.r else 0
            tlc = np.random.randint(0, cols - self.c) if cols > self.c else 0

            if self.data_format == 'channels_first':
                x *= bg[:, tlr:tlr + self.r, tlc:tlc + self.c]
            elif self.data_format == 'channels_last':
                x *= bg[tlr:tlr + self.r, tlc:tlc + self.c, :]
            x /= 255

            params = self.image_data_generator.get_random_transform(x.shape)
            bX[i] = self.image_data_generator.standardize(
                self.image_data_generator.apply_transform(x, params))
            if 'mask' in self.y:
                by['mask'][i] = self.image_data_generator.standardize(
                    self.image_data_generator.apply_transform(mask, params))
            by['p'][i] = np_utils.to_categorical(np.asarray([y]),
                                                 len(self.y['p'][0]))

        permutation = np.random.permutation(len(bX))
        bX = bX[permutation]
        for k in self.y.keys():
            by[k] = by[k][permutation]
        return bX, by

    def __len__(self):
        return int(np.ceil(self.samples / float(self.batch_size)))
