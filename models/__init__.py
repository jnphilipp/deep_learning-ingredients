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

from .core import ingredient, get, load, log_param_count, save, summary, plot

# from . import autoencoder
from . import cnn
from . import dense

# from . import gan
from . import rnn

# from . import siamese

__all__ = (
    "ingredient",
    "get",
    "load",
    "log_param_count",
    "save",
    "summary",
    "plot",
    "cnn",
    "dense",
    "rnn",
)
