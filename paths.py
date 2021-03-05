# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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
"""Paths ingredient."""

import os

from sacred import Ingredient
from typing import Iterable


ingredient = Ingredient("paths")


@ingredient.config
def config():
    """Default config."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(project_dir, "datasets")


@ingredient.capture
def join(*args) -> str:
    """Join to a path and replace escaped configs."""
    return _join(args)


@ingredient.capture
def _join(parts: Iterable[str], project_dir, datasets_dir) -> str:
    return os.path.join(*[part.format(datasets_dir=datasets_dir) for part in parts])
