# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
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
"""RNN module for models ingredient."""

from tensorflow.keras.layers import Bidirectional, Layer
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Callable, Dict, List, Optional

from .ingredient import ingredient


@ingredient.capture(prefix="layers")
def block(
    nb_layers: int,
    recurrent: Dict,
    bidirectional: Optional[Dict] = None,
    **kwargs,
) -> Callable:
    """Create RNN block from config."""
    _layers: Dict[int, Layer] = {}
    for i in range(nb_layers):
        rnn_layer = dict(**recurrent)
        if i != nb_layers - 1:
            rnn_layer["config"] = dict(
                rnn_layer["config"], **{"return_sequences": True}
            )

        if bidirectional:
            _layers[i] = Bidirectional.from_config(
                dict(bidirectional, **{"layer": rnn_layer})
            )
        else:
            _layers[i] = deserialize_layer(rnn_layer)

    def _block(tensors: List[KerasTensor]) -> List[KerasTensor]:
        xs = [None for i in tensors]
        for i in range(nb_layers):
            for j, x in enumerate(xs):
                xs[j] = _layers[i](tensors[j] if x is None else x)
        return xs

    return _block
