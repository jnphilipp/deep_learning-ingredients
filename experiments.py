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

from logging import Logger
from sacred import Ingredient
from sacred.run import Run
from tensorflow.keras.models import Model
from typing import Sequence, Union

from ingredients import history as history_ingredient
from ingredients import models as models_ingredient


ingredient = Ingredient('experiments',
                        ingredients=[history_ingredient.ingredient,
                                     models_ingredient.ingredient])


@ingredient.config
def config():
    id = None


@ingredient.capture
def save(models: Union[Model, Sequence[Model]], history: dict, _log: Logger,
         _run: Run):
    _log.info('Saving experiment.')
    path = _run.observers[0].run_dir

    try:
        for model in models:
            models_ingredient.save(model, model.name, path)
            history_ingredient.save(f'{model.name}-train_history', history,
                                    path)
    except TypeError:
        models_ingredient.save(models, path=path)
        history_ingredient.save('train_history', history, path)
