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

from keras.layers import (add, average, concatenate, dot, maximum, minimum,
                          multiply, subtract)
from keras.layers import Embedding, Input
from keras.layers import deserialize as deserialize_layer
from tensorflow import Tensor
from typing import Iterable

from . import ingredient


@ingredient.capture(prefix='merge')
def merge_layer(inputs: Iterable[Tensor], t: str, config: dict = {}):
    if t == 'add':
        return add(inputs, **config)
    elif t == 'average':
        return average(inputs, **config)
    elif t == 'concatenate':
        return concatenate(inputs, **config)
    elif t == 'dot':
        return dot(inputs, **config)
    elif t == 'maximum':
        return maximum(inputs, **config)
    elif t == 'minimum':
        return minimum(inputs, **config)
    elif t == 'multiply':
        return multiply(inputs, **config)
    elif t == 'subtract':
        return subtract(inputs, **config)
