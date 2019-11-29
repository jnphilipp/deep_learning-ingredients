# -*- coding: utf-8 -*-
# Copyright (C) 2019 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of deep_learning-ingredients.
#
# deep_learning-ingredients is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# deep_learning-ingredients is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with deep_learning-ingredients. If not, see
# <http://www.gnu.org/licenses/>.

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
