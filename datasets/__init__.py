# -*- coding: utf-8 -*-

import os

from sacred import Ingredient
ingredient = Ingredient('datasets')


from . import h5py
from . import images
from . import json
from . import keras
from . import text
from .core import *
from .. import PROJECT_DIR


@ingredient.config
def config():
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
