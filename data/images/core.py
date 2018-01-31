# -*- coding: utf-8 -*-

import csv
import numpy as np
import os

from ingredients.data import ingredients
from keras import backend as K
from keras.preprocessing.image import img_to_array, list_pictures, load_img
from keras.utils import np_utils
from scipy import ndimage


@ingredients.config
def config():
    ext = 'jpg|jpeg|bmp|png'
    grayscale = False
    masks = False
    load_images = True
    X_fields = []
    y_fields = []


@ingredients.capture
def load_img_array(path, grayscale):
    return img_to_array(load_img(path, grayscale))


@ingredients.capture
def load(DATASETS_DIR, dataset, which_set, ext, grayscale, masks, load_images,
         X_fields=[], y_fields=[], **kwargs):
    print('Loading images [dataset=%s - which_set=%s]...' % (dataset,
                                                             which_set))

    dataset_path = os.path.join(DATASETS_DIR, dataset, which_set)
    if dataset_path.endswith('.csv'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            has_header = kwargs['has_header'] if 'has_header' in kwargs else True
            if has_header:
                fields = reader.fieldnames
                print(fields)

            nb_classes = {}
            X = []
            y = []
            for row in reader:
                if X_fields:
                    tmp = {}
                    for field in X_fields:
                        if not row[field]:
                            continue

                        path = os.path.join(DATASETS_DIR, dataset, row[field])
                        if load_images:
                            tmp[field] = load_img_array(path)
                        else:
                            tmp[field] = path
                    X.append(tmp)
                else:
                    X.append(row)

                for field in y_fields:
                    if field not in nb_classes:
                        nb_classes[field] = set()
                    nb_classes[field].add(row[field])
                y.append({field:row[field] for field in y_fields})
        nb = {k:len(v) for k, v in nb_classes.items()}
    elif os.path.isdir(dataset_path):
        classes = {}
        nb_classes = 0
        for e in sorted(os.scandir(dataset_path), key=lambda e: e.name):
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
            if masks and 'mask' in os.path.basename(picture):
                continue
            name = os.path.basename(picture)
            if masks:
                mask_picture = os.path.join(os.path.dirname(picture),
                                            name.replace('.png', '.mask.png'))
                mask_name = os.path.basename(mask_picture)
                if os.path.exists(mask_picture):
                    X.append(load_img_array(picture) if load_images else picture)
                    Xmasks.append(load_img_array(mask_picture, True)
                        if load_images else mask_picture)
                    names.append((name, mask_name))
                else:
                    continue
            else:
                X.append(load_img_array(picture) if load_images else picture)
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
def speckle(img):
    """This creates larger "blotches" of noise which look more realistic than
    just adding gaussian noise assumes greyscale with pixels ranging from 0 to 1
    """
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck