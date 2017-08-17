# -*- coding: utf-8 -*-

import numpy as np
import os

from csv import DictReader
from ingredients.data import ingredients
from keras import backend as K
from keras.preprocessing.image import img_to_array, list_pictures, load_img
from keras.utils import np_utils


@ingredients.capture
def sentences(DATASETS_DIR, dataset, which_set, filters=None, clean=None):
    print('Loading sentences [dataset=%s - which_set=%s]...' % (dataset,
                                                                which_set))
    sentences = []
    for file in os.listdir(os.path.join(DATASETS_DIR, dataset, which_set)):
        with open(os.path.join(DATASETS_DIR, dataset, which_set, file), 'r',
                  encoding='utf-8') as f:
            for line in f:
                if filters is not None:
                    if filters(line):
                        if clean is not None:
                            sentences.append(clean(line.strip()))
                        else:
                            sentences.append(line.strip())
                else:
                    if clean is not None:
                        sentences.append(clean(line.strip()))
                    else:
                        sentences.append(line.strip())
    return sentences
