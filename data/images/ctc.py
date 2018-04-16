# -*- coding: utf-8 -*-

import cairocffi as cairo
import math
import numpy as np

from ingredients.data import ingredients
from keras import backend as K
from keras.preprocessing.image import random_rotation

from .patch import patch


@ingredients.config
def config():
    fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L',
             'FreeMono']
    rg = 90


@ingredients.capture
def paint_text(text, height, width, fonts, font_size, font_rgb, rg):
    """Paints the string in a random location in a random font, with a slight
    random rotation
    """
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1.0, 1.0, 1.0)
        context.paint()
        context.select_font_face(
            np.random.choice(fonts),
            np.random.choice([cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC,
                              cairo.FONT_SLANT_OBLIQUE]),
            np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL])
        )
        context.set_font_size(font_size)
        box = context.text_extents(text)

        max_shift_x = (width // 3) - abs(box[0])
        top_left_x = min(max(np.random.normal(max_shift_x // 2,
                                              max(max_shift_x // 4, 0.001)),
                             abs(box[0])), max_shift_x)

        max_shift_y = height - abs(box[1])
        top_left_y = min(max(np.random.normal(max_shift_y // 2,
                                              max(max_shift_y // 4, 0.001)),
                             abs(box[1])), height)

        context.move_to(top_left_x, top_left_y)
        context.set_source_rgb(font_rgb[0], font_rgb[1], font_rgb[2])
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (height, width, 4)
    if K.image_data_format() == 'channels_first':
        a = a[0:3, :, :]
        a = a.transpose(2, 0, 1)
        channel_axis = 0
        row_axis = 1
        col_axis = 2
    elif K.image_data_format() == 'channels_last':
        a = a[:, :, 0:3]
        channel_axis = 2
        row_axis = 0
        col_axis = 1

    return random_rotation(a.astype(np.float32) / 255,
                           (-rg - rg) * np.random.random() + rg,
                           row_axis=row_axis, col_axis=col_axis,
                           channel_axis=channel_axis)


def get_generator(backgrounds, text_generator, font_size, font_rgb):
    def generator(N, shape, mask_shape, batch_size=128):
        if K.image_data_format() == 'channels_first':
            r = shape[1]
            c = shape[2]
        elif K.image_data_format() == 'channels_last':
            r = shape[0]
            c = shape[1]

        b = 0
        while True:
            current_index = (b * batch_size) % N
            current_batch_size = batch_size if N >= current_index + batch_size \
                else N - current_index
            b = b + 1 if current_batch_size == batch_size else 0

            bX = np.zeros((current_batch_size,) + tuple(shape))
            bmasks = np.zeros((current_batch_size,) + mask_shape)
            for i in range(current_batch_size):
                text_img = paint_text(text_generator(), r, c,
                                      font_size=font_size(),
                                      font_rgb=font_rgb())
                bX[i] = text_img * patch(
                    np.random.choice(backgrounds) / 255, r, c
                )[0]
                if K.image_data_format() == 'channels_first':
                    bmasks[i] = np.dot(
                        text_img[:3,...], [0.299, 0.587, 0.114]
                    ).reshape(mask_shape)
                elif K.image_data_format() == 'channels_last':
                    bmasks[i] = np.dot(
                        text_img[...,:3], [0.299, 0.587, 0.114]
                    ).reshape(mask_shape)

            bmasks[bmasks < .9] = 0.
            bmasks[bmasks >= .9] = 1.
            bmasks = 1 - bmasks
            yield bX, bmasks

    return generator
