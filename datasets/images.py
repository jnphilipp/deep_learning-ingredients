# -*- coding: utf-8 -*-

import numpy as np
import os
import re

from keras import backend as K
from keras.preprocessing.image import image
from keras.utils import np_utils
from scipy import ndimage


from .. import ingredient


@ingredient.capture
def load(path, color_mode='rgb', target_size=None, interpolation='nearest',
         data_format=None, dtype='float32'):

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    img = image.load_img(path, False, color_mode, target_size, interpolation)
    return image.img_to_array(img, data_format, dtype)


@ingredient.capture
def from_directory(DATASETS_DIR, dataset, which_set,
                   ext=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif', 'tiff'),
                   _log=None):
    _log.info(f'Loading images [{dataset}: {which_set}].')
    dataset_path = os.path.join(DATASETS_DIR, dataset, which_set)
    mask_ext = tuple(f'.mask.{e}' for e in ((ext,)
                           if isinstance(ext, str) else ext))

    X = []
    y = {'p': [], 'mask': []}
    num_classes = 0
    for e in sorted(os.scandir(dataset_path), key=lambda e: e.name):
        if e.is_dir():
            for img in image.list_pictures(e.path, ext):
                if img.lower().endswith(mask_ext):
                    continue
                X.append(load(img))
                y['p'].append(num_classes)

                mask_img = re.sub(rf'.({"|".join(ext)})$', r'.mask\g<0>', img)
                if os.path.exists(mask_img):
                    y['mask'].append(load(mask_img))

            num_classes += 1

    samples = len(X)
    y['p'] = np_utils.to_categorical(np.asarray(y['p']), num_classes)
    _log.info(f'Found {samples} images belonging to {num_classes} classes.')

    if len(y['mask']) != samples:
        y = y['p']

    return X, y


@ingredient.capture
def speckle(x, data_format=None):
    """This creates larger "blotches" of noise which look more realistic than
    just adding gaussian noise assumes pixels ranging from 0 to 1
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    if data_format == 'channels_first':
        r = x.shape[1]
        c = x.shape[2]
        blur = np.random.rand(1, r, c)
    elif data_format == 'channels_last':
        r = x.shape[0]
        c = x.shape[1]
        blur = np.random.rand(r, c, 1)

    severity = np.random.uniform(0, 0.7)
    speck = ndimage.gaussian_filter(blur * severity, 1)
    speck[speck > 1.] = 1.
    speck[speck <= 0.] = 0.
    return x * speck
