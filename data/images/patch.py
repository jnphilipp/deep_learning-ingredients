# -*- coding: utf-8 -*-

import numpy as np

from ingredients.data import ingredients
from keras import backend as K


@ingredients.capture
def patch(x, height, width, *args, **kwargs):
    r = x.shape[1 if K.image_data_format() == 'channels_first' else 0]
    c = x.shape[2 if K.image_data_format() == 'channels_first' else 1]
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