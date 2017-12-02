# -*- coding: utf-8 -*-

import os

from sacred import Ingredient
ingredients = Ingredient('data')


from ingredients import PROJECT_DIR
from .core import *
from . import h5py
from . import images
from . import json
from . import predict
from . import text


@ingredients.config
def config():
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
