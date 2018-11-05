# -*- coding: utf-8 -*-

import os

from sacred import Ingredient
ingredient = Ingredient('datasets')


from .. import PROJECT_DIR


@ingredient.config
def config():
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')


@ingredient.capture
def get_full_dataset_path(DATASETS_DIR, dataset, train_set,
                          validation_set=None):
    base_dir = os.path.join(DATASETS_DIR, dataset)
    trainset_path = os.path.join(base_dir, train_set)
    if validation_set:
        return trainset_path, os.path.join(base_dir, validation_set)
    else:
        return trainset_path, None


from . import h5py
from . import images
from . import json
from . import keras
from . import text
from .core import *
