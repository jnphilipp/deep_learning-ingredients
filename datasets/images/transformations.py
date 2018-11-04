# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
from scipy import ndimage

from .. import ingredient


@ingredient.capture
def speckle(x):
    """This creates larger "blotches" of noise which look more realistic than
    just adding gaussian noise assumes pixels ranging from 0 to 1
    """
    if K.image_data_format() == 'channels_first':
        r = x.shape[1]
        c = x.shape[2]
        blur = np.random.rand(1, r, c)
    elif K.image_data_format() == 'channels_last':
        r = x.shape[0]
        c = x.shape[1]
        blur = np.random.rand(r, c, 1)

    severity = np.random.uniform(0, 0.7)
    speck = ndimage.gaussian_filter(blur * severity, 1)
    speck[speck > 1.] = 1.
    speck[speck <= 0.] = 0.
    return x * speck
