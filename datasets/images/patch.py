# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from .. import ingredient


@ingredient.capture
def patch(x, height, width, *args, **kwargs):
    if K.image_data_format() == 'channels_first':
        r = x.shape[1]
        c = x.shape[2]
    elif K.image_data_format() == 'channels_last':
        r = x.shape[0]
        c = x.shape[1]

    top_left_x = np.random.randint(0, r - height) if r > height else 0
    top_left_y = np.random.randint(0, c - width) if c > width else 0

    values = []
    if K.image_data_format() == 'channels_first':
        values.append(x[:, top_left_x:top_left_x + height,
                        top_left_y:top_left_y + width])
        for a in args:
            values.append(a[:, top_left_x:top_left_x + height,
                            top_left_y:top_left_y + width])
        for k in kwargs.keys():
            values.append(kwargs[k][:, top_left_x:top_left_x + height,
                                    top_left_y:top_left_y + width])
    else:
        values.append(x[top_left_x:top_left_x + height,
                        top_left_y:top_left_y + width, :])
        for a in args:
            values.append(a[top_left_x:top_left_x + height,
                            top_left_y:top_left_y + width, :])
        for k in kwargs.keys():
            values.append(kwargs[k][top_left_x:top_left_x + height,
                                    top_left_y:top_left_y + width, :])
    return tuple(values)
