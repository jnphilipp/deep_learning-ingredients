# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2019-2022 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of deep_learning-ingredients.
#
# deep_learning-ingredients is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# deep_learning-ingredients is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with deep_learning-ingredients. If not, see <http://www.gnu.org/licenses/>.
"""Experiment ingredient."""

from logging import Logger
from sacred import Ingredient
from sacred.run import Run
from tensorflow.keras.models import Model
from typing import Dict, Sequence, Union

from ingredients import history as history_ingredient
from ingredients import models as models_ingredient


ingredient = Ingredient(
    "experiments",
    ingredients=[history_ingredient.ingredient, models_ingredient.ingredient],
)


@ingredient.capture
def save(
    path: str,
    models: Union[Model, Sequence[Model]],
    history: Dict,
    _log: Logger,
    _run: Run,
):
    """Save an experiment."""
    _log.info("Saving experiment.")

    try:
        for model in models:
            models_ingredient.save(model, model.name, path)
            history_ingredient.save(f"{model.name}-train_history", history, path)
    except TypeError:
        models_ingredient.save(models, path=path)
        history_ingredient.save("train_history", history, path)
