# -*- coding: utf-8 -*-

from ingredients import models
from sacred import Ingredient
ingredients = Ingredient('experiment', ingredients=[models.ingredients])


import os

from ingredients import PROJECT_DIR
from ingredients.experiment import history


@ingredients.config
def config():
    experiments_dir = os.path.join(PROJECT_DIR, 'experiments')
    id = None


@ingredients.capture
def save(path, model, train_history=None):
    try:
        for m in model:
            models.save(path, m, m.name)
    except TypeError:
        models.save(path, model)
    if train_history:
        history.save(path, 'train_history', train_history)
