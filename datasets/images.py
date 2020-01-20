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

import numpy as np
import os
import re

from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from logging import Logger
from scipy import ndimage
from typing import Dict, List, Optional, Sequence, Tuple, Union


from . import ingredient


@ingredient.capture
def list_pictures(directory: str,
                  ext: Union[str, Sequence[str]] = ('jpg', 'jpeg', 'bmp',
                                                    'png', 'ppm', 'tif',
                                                    'tiff')) -> List[str]:
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]


@ingredient.capture
def load(path: str, color_mode: str = 'rgb',
         target_size: Optional[Tuple] = None, interpolation: str = 'nearest',
         dtype: str = 'float32') -> np.ndarray:
    return img_to_array(load_img(path, False, color_mode, target_size,
                                 interpolation), dtype=dtype)


@ingredient.capture
def from_directory(path: str, _log: Logger, mask_names: List[str] = ['mask'],
                   ext: Union[str, Sequence[str]] = ('jpg', 'jpeg', 'bmp',
                                                     'png', 'ppm', 'tif',
                                                     'tiff')) -> \
        Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    def load_images(path: str):
        for img in list_pictures(path, ext):
            if img.lower().endswith(mask_ext):
                continue
            X.append(load(img))
            y['p'].append(nb_classes)

            for mask_name in mask_names:
                mask_img = re.sub(rf'.({"|".join(ext)})$',
                                  rf'.{mask_name}\g<0>', img)
                if os.path.exists(mask_img):
                    y[mask_name].append(load(mask_img, 'grayscale'))

    _log.info(f'Loading images from {path}.')

    mask_ext = tuple(f'.{mask_name}.{e}' for mask_name in mask_names
                     for e in ((ext,) if isinstance(ext, str) else ext))

    X: List[np.ndarray] = []
    y: Dict[str, np.ndarray] = {mask_name: []
                                for mask_name in mask_names + ['p']}
    nb_classes = 0
    for e in sorted(os.scandir(path), key=lambda e: e.name):
        if e.is_dir():
            load_images(e.path)
            nb_classes += 1
    if nb_classes == 0 and len(X) == 0:
        load_images(path)

    samples = len(X)
    if nb_classes > 0:
        y['p'] = to_categorical(np.asarray(y['p']), nb_classes)
    else:
        del y['p']

    for mask_name in mask_names:
        if len(y[mask_name]) != samples:
            del y[mask_name]

    _log.info(f'Found {samples} images belonging to {nb_classes} classes and' +
              f' with masks {", ".join(k for k in y.keys() if k != "p")}.')

    return X, y


@ingredient.capture
def speckle(x: np.ndarray, severity: float = np.random.uniform(0, 0.7)) -> \
        np.ndarray:
    """This creates larger "blotches" of noise which look more realistic than
    just adding gaussian noise assumes pixels ranging from 0 to 1
    """
    blur = np.random.rand(x.shape[0], x.shape[1], 1)
    speck = ndimage.gaussian_filter(blur * severity, 1)
    speck[speck > 1.] = 1.
    speck[speck <= 0.] = 0.
    return x * speck
