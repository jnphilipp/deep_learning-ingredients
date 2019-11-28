# -*- coding: utf-8 -*-

import os

from sacred import Ingredient
ingredient = Ingredient('datasets')


from .. import PROJECT_DIR


@ingredient.config
def config():
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')


from . import keras
