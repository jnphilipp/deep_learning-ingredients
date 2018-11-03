# -*- coding: utf-8 -*-

import os

from sacred import Ingredient

from . import PROJECT_DIR
from . import history as history_ingredient
from . import models as models_ingredient


ingredient = Ingredient('experiments',
                        ingredients=[history_ingredient.ingredient,
                                     models_ingredient.ingredient])


@ingredient.config
def config():
    base_dir = os.path.join(PROJECT_DIR, 'experiments')
    id = None


@ingredient.capture
def save(models, train_history=None):
    try:
        for model in models:
            models_ingredient.save(model, model.name)
            if train_history:
                history_ingredient.save('%s-train_history' % model.name,
                                        train_history)
    except TypeError:
        models_ingredient.save(models)
        if train_history:
            history_ingredient.save('train_history', train_history)
