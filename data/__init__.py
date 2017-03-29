# -*- coding: utf-8 -*-

from sacred import Ingredient
ingredients = Ingredient('data')


import os

from ingredients import PROJECT_DIR
from . import images


@ingredients.config
def config():
    DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
