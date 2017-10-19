# -*- coding: utf-8 -*-

from ingredients import models as models_ingredients
from sacred import Ingredient
ingredients = Ingredient('experiments',
                         ingredients=[models_ingredients.ingredients])


import os

from ingredients import PROJECT_DIR
from ingredients.experiments import history
from ingredients.experiments import plots


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
