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
"""Dense module for models ingredient."""

from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Layer
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Callable, Dict, List, Optional

from .ingredient import ingredient


@ingredient.capture(prefix="layers")
def block(
    nb_layers: int,
    dense: Dict,
    batchnorm: Optional[Dict] = None,
    dropout: Optional[Dict] = None,
    **kwargs,
) -> Callable:
    """Create Dense block from config."""
    _layers: Dict[int, Layer] = {}
    for i in range(nb_layers):
        dense_layer = dict(**dense)
        if "output_shape" in kwargs and i == nb_layers - 1:
            dense_layer["units"] = kwargs["output_shape"]
        elif "units" in dense and type(dense["units"]) == list:
            dense_layer["units"] = dense["units"][i]

        if batchnorm and dropout:
            activation = dense_layer["activation"]
            dense_layer["activation"] = "linear"
            _layers[i] = (
                Dense.from_config(dense_layer),
                BatchNormalization.from_config(batchnorm),
                Activation(activation),
                deserialize_layer(dropout),
            )
        elif batchnorm and not dropout:
            activation = dense_layer["activation"]
            dense_layer["activation"] = "linear"
            _layers[i] = (
                Dense.from_config(dense_layer),
                BatchNormalization.from_config(batchnorm),
                Activation(activation),
            )
        elif not batchnorm and dropout:
            _layers[i] = (Dense.from_config(dense_layer), deserialize_layer(dropout))
        else:
            _layers[i] = Dense.from_config(dense_layer)

    def _block(tensors: List[KerasTensor]) -> List[KerasTensor]:
        xs = [None for i in tensors]
        for i in range(nb_layers):
            for j, x in enumerate(xs):
                if batchnorm and dropout:
                    xs[j] = _layers[i][3](
                        _layers[i][2](
                            _layers[i][1](_layers[i][0](tensors[j] if x is None else x))
                        )
                    )
                elif batchnorm and not dropout:
                    xs[j] = _layers[i][2](
                        _layers[i][1](_layers[i][0](tensors[j] if x is None else x))
                    )
                elif not batchnorm and dropout:
                    xs[j] = _layers[i][1](_layers[i][0](tensors[j] if x is None else x))
                else:
                    xs[j] = _layers[i](tensors[j] if x is None else x)
        return xs

    return _block
