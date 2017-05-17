# -*- coding: utf-8 -*-

import numpy as np
import os

from csv import DictReader
from ingredients.data import ingredients
from keras import backend as K
from keras.preprocessing.image import img_to_array, list_pictures, load_img
from keras.utils import np_utils


@ingredients.config
def config():
    ext = 'jpg|jpeg|bmp|png'
    grayscale = False
    masks = False


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, ext, grayscale, masks):
    print('Loading images [dataset=%s - which_set=%s]...' % (dataset, which_set))

    classes = {}
    nb_classes = 0
    if os.path.exists(os.path.join(DATASETS_DIR, dataset, which_set, 'classes.csv')):
        with open(os.path.join(DATASETS_DIR, dataset, which_set, 'classes.csv'),
                  'r', encoding='utf8') as f:
            reader = DictReader(f)
            for row in reader:
                classes[row['image']] = [int(c) for c in row['class'].split(',')] \
                    if ',' in row['class'] else [int(row['class'])]
        nb_classes = len(set([c for img in classes.values() for c in img]))
    else:
        for e in sorted(os.scandir(os.path.join(DATASETS_DIR, dataset, which_set)),
                        key=lambda e: e.name):
            if e.is_dir():
                for p in list_pictures(e.path):
                    if masks and 'mask' in os.path.basename(p):
                        continue
                    classes[os.path.basename(p)] = [nb_classes]
                nb_classes += 1

    X = []
    y = []
    Xmasks = []
    names = []
    for picture in sorted(list_pictures(os.path.join(DATASETS_DIR,
                                        dataset, which_set), ext=ext)):
        if masks and 'mask' in os.path.basename(p):
            continue
        name = os.path.basename(picture)
        if masks:
            mask_picture = os.path.join(os.path.dirname(picture),
                                        name.replace('.png', '.mask.png'))
            if os.path.exists(mask_picture):
                X.append(img_to_array(load_img(picture, grayscale)))
                Xmasks.append(img_to_array(load_img(mask_picture, True)))
            else:
                continue
        else:
            X.append(img_to_array(load_img(picture, grayscale)))
        names.append(name)
        if os.path.basename(picture) in classes:
            y.append(classes[os.path.basename(picture)])
    if len(set([x.shape for x in X])) <= 1:
        X = np.asarray(X)
        if masks:
            Xmasks = np.asarray(Xmasks)
    if y:
        if max([type(i) == list for i in y]) and max([len(i) > 1 for i in y]):
            max_len = max([len(i) for i in y])
            new_y = np.zeros((len(y), max_len, nb_classes))
            for i, classes in enumerate(y):
                for j, c in enumerate(classes):
                    new_y[i, j, c] = 1
            y = new_y
        else:
            y = np_utils.to_categorical(np.asarray(y), nb_classes)
        if masks:
            return X, y, Xmasks, names, nb_classes
        else:
            return X, y, names, nb_classes
    else:
        if masks:
            return X, Xmasks, names
        else:
            return X, names


@ingredients.capture
def patch(x, height, width, *args, **kwargs):
    r = x.shape[1 if K.image_data_format() == 'channels_first' else 0]
    c = x.shape[2 if K.image_data_format() == 'channels_first' else 1]
    top_left_x = np.random.randint(0, r - height) if r > height else 0
    top_left_y = np.random.randint(0, c - width) if c > width else 0

    values = []
    if K.image_data_format() == 'channels_first':
        values.append(x[0:x.shape[0],
                        top_left_x:top_left_x + height,
                        top_left_y:top_left_y + width])
        for a in args:
            values.append(a[0:x.shape[0],
                            top_left_x:top_left_x + height,
                            top_left_y:top_left_y + width])
        for k in kwargs.keys():
            values.append(kwargs[k][0:x.shape[0],
                                    top_left_x:top_left_x + height,
                                    top_left_y:top_left_y + width])
    else:
        values.append(x[top_left_x:top_left_x + height,
                        top_left_y:top_left_y + width,
                        0:x.shape[2]])
        for a in args:
            values.append(a[top_left_x:top_left_x + height,
                            top_left_y:top_left_y + width,
                            0:x.shape[2]])
        for k in kwargs.keys():
            values.append(kwargs[k][top_left_x:top_left_x + height,
                                    top_left_y:top_left_y + width,
                                    0:x.shape[2]])
    return tuple(values)
