# -*- coding: utf-8 -*-

from ingredients import models as models_ingredients
from sacred import Ingredient
ingredients = Ingredient('experiment',
                         ingredients=[models_ingredients.ingredients])


import os

from ingredients import PROJECT_DIR
from ingredients.experiment import history


@ingredients.config
def config():
    experiments_dir = os.path.join(PROJECT_DIR, 'experiments')
    id = None


@ingredients.capture
def save(path, models, train_history=None):
    try:
        for model in models:
            models_ingredients.save(path, model, model.name)
    except TypeError:
        models_ingredients.save(path, models)
    if train_history:
        history.save(path, 'train_history', train_history)
