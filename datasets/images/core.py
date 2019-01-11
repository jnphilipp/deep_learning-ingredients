# -*- coding: utf-8 -*-

import csv
import numpy as np
import os
import re

from keras import backend as K
from keras.preprocessing.image import image
from keras.utils import np_utils


from .. import ingredient


# @ingredient.config
# def config():
#     ext = 'jpg|jpeg|bmp|png|ppm'
#     grayscale = False
#     masks = []
#     load_images = True
#     rescale = 1./255

#     X_fields = []
#     y_fields = []


@ingredient.capture
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    for root, _, files in os.walk(directory):
        for f in files:
            if re.match(r'([\w\.-]+\.(?:' + ext + '))', f):
                yield os.path.join(root, f)


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
                   ext='jpg|jpeg|bmp|png|ppm', _log=None):
    _log.info(f'Loading images [{dataset}: {which_set}].')
    dataset_path = os.path.join(DATASETS_DIR, dataset, which_set)

    X = []
    y = []
    num_classes = 0
    for e in sorted(os.scandir(dataset_path), key=lambda e: e.name):
        if e.is_dir():
            for img in image.list_pictures(e.path, ext):
                X.append(load(img))
                y.append(num_classes)
            num_classes += 1

    samples = len(X)
    y = np_utils.to_categorical(np.asarray(y), num_classes)
    _log.info(f'Found {samples} images belonging to {num_classes} classes.')

    return X, y


# @ingredient.capture
# def load(DATASETS_DIR, dataset, which_set, ext, grayscale, masks, load_images,
#          _log, X_fields=[], y_fields=[], **kwargs):
#     _log.info('Loading images [%s: %s].' % (dataset, which_set))

#     dataset_path = os.path.join(DATASETS_DIR, dataset, which_set)
#     if dataset_path.endswith('.csv'):
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)

#             if 'has_header' in kwargs:
#                 has_header = kwargs['has_header']
#             else:
#                 has_header = True

#             if has_header:
#                 fields = reader.fieldnames
#                 _log.info(fields)

#             nb_classes = {}
#             X = []
#             y = []
#             for row in reader:
#                 if X_fields:
#                     tmp = {}
#                     for field in X_fields:
#                         if not row[field]:
#                             continue

#                         path = os.path.join(DATASETS_DIR, dataset, row[field])
#                         if load_images:
#                             tmp[field] = load_img(path)
#                         else:
#                             tmp[field] = path
#                     X.append(tmp)
#                 else:
#                     X.append(row)

#                 for field in y_fields:
#                     if field not in nb_classes:
#                         nb_classes[field] = set()
#                     nb_classes[field].add(row[field])
#                 y.append({field: row[field] for field in y_fields})
#         nb = {k: len(v) for k, v in nb_classes.items()}
#     elif os.path.isdir(dataset_path):
#         classes = {}
#         nb_classes = 0
#         for e in sorted(os.scandir(dataset_path), key=lambda e: e.name):
#             if e.is_dir():
#                 for p in list_pictures(e.path):
#                     if masks and 'mask' in os.path.basename(p):
#                         continue
#                     classes[os.path.basename(p)] = [nb_classes]
#                 nb_classes += 1

#         X = []
#         y = []
#         Xmasks = []
#         names = []
#         for picture in sorted(list_pictures(os.path.join(DATASETS_DIR,
#                                             dataset, which_set), ext=ext)):
#             if masks and 'mask' in os.path.basename(picture):
#                 continue
#             name = os.path.basename(picture)
#             if masks:
#                 mask_picture = os.path.join(os.path.dirname(picture),
#                                             name.replace('.png', '.mask.png'))
#                 mask_name = os.path.basename(mask_picture)
#                 if os.path.exists(mask_picture):
#                     X.append(load_img(picture) if load_images else picture)
#                     Xmasks.append(load_img(mask_picture, True)
#                                   if load_images else mask_picture)
#                     names.append((name, mask_name))
#                 else:
#                     continue
#             else:
#                 X.append(load_img(picture) if load_images else picture)
#                 names.append(name)
#             if os.path.basename(picture) in classes:
#                 y.append(classes[os.path.basename(picture)])

#         if len(set([x.shape for x in X])) <= 1:
#             X = np.asarray(X)
#             if masks:
#                 Xmasks = np.asarray(Xmasks)

#     if y:
#         if max([type(i) == list for i in y]) and max([len(i) > 1 for i in y]):
#             max_len = max([len(i) for i in y])
#             new_y = np.zeros((len(y), max_len, nb_classes))
#             for i, classes in enumerate(y):
#                 for j, c in enumerate(classes):
#                     new_y[i, j, c] = 1
#             y = new_y
#         else:
#             y = np_utils.to_categorical(np.asarray(y), nb_classes)
#         if masks:
#             return X, y, Xmasks, names, nb_classes
#         else:
#             return X, y, names, nb_classes
#     else:
#         if masks:
#             return X, Xmasks, names
#         else:
#             return X, names
