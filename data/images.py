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


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, ext, grayscale):
    print('Loading images [dataset=%s - which_set=%s]...' % (dataset, which_set))

    classes = {}
    nb_classes = 0
    if os.path.exists(os.path.join(DATASETS_DIR, dataset, which_set, 'classes.csv')):
        with open(os.path.join(DATASETS_DIR, dataset, which_set, 'classes.csv'), 'r', encoding='utf8') as f:
            reader = DictReader(f)
            for row in reader:
                classes[row['image']] = [int(c) for c in row['class'].split(',')] if ',' in row['class'] else [int(row['class'])]
        nb_classes = len(set([c for img in classes.values() for c in img]))
    else:
        for e in sorted(os.scandir(os.path.join(DATASETS_DIR, dataset, which_set)), key=lambda e: e.name):
            if e.is_dir():
                for p in list_pictures(e.path):
                    classes[os.path.basename(p)] = [nb_classes]
                nb_classes += 1

    X = []
    y = []
    names = []
    for picture in sorted(list_pictures(os.path.join(DATASETS_DIR, dataset, which_set), ext=ext)):
        X.append(img_to_array(load_img(picture, grayscale)))
        names.append(os.path.basename(picture))
        if os.path.basename(picture) in classes:
            y.append(classes[os.path.basename(picture)])
    if len(set([x.shape for x in X])) <= 1:
        X = np.asarray(X)
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
        return X, y, names, nb_classes
    else:
        return X, names


@ingredients.capture
def patch(x, height, width):
    top_left_x = np.random.randint(0, x.shape[1 if K.image_data_format() == 'channels_first' else 0] - height) if x.shape[1 if K.image_data_format() == 'channels_first' else 0] > height else 0
    top_left_y = np.random.randint(0, x.shape[2 if K.image_data_format() == 'channels_first' else 1] - width) if x.shape[2 if K.image_data_format() == 'channels_first' else 1] > width else 0
    if K.image_data_format() == 'channels_first':
        return x[0:x.shape[0], top_left_x:top_left_x + height, top_left_y:top_left_y + width]
    else:
        return x[top_left_x:top_left_x + height, top_left_y:top_left_y + width, 0:x.shape[2]]
