# -*- coding: utf-8 -*-

from ingredients import model
from sacred import Ingredient
ingredients = Ingredient('experiment', ingredients=[model.ingredients])


import os

from ingredients import PROJECT_DIR
from ingredients.experiment import history


@ingredients.config
def config():
    experiments_dir = os.path.join(PROJECT_DIR, 'experiments')
    id = None


@ingredients.capture
def save(path, m, train_history=None):
    try:
        for m in model:
            model.save(path, m, m.name)
    except TypeError:
        model.save(path, model)
    if train_history:
        history.save(path, 'train_history', train_history)
