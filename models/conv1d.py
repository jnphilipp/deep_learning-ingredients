# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2019-2022 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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
"""Conv1d module for models ingredient."""

from tensorflow.keras.layers import Concatenate, Conv1D, Layer
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Callable, Dict, List, Optional, Union

from .ingredient import ingredient


@ingredient.capture(prefix="layers")
def block(
    nb_layers: int,
    connection_type: str = "base",
    conv1d: Dict = {},
    concat_axis: int = -1,
    filters: Optional[int] = None,
    **kwargs,
) -> Callable:
    """Get conv1d block from config."""
    assert connection_type in ["base", "densely"]
    if connection_type == "densely":
        assert concat_axis is not None

    _layers: Dict[int, Layer] = {}
    for i in range(nb_layers):
        if connection_type == "base":
            _layers[i] = layer(filters=filters)
        elif connection_type == "densely":
            _layers[i] = (layer(densley=True), Concatenate(axis=concat_axis))

    def _block(
        tensors: Union[KerasTensor, List[KerasTensor]]
    ) -> Union[KerasTensor, List[KerasTensor]]:
        _densely = (
            [[tensors]] if not isinstance(tensors, list) else [[i] for i in tensors]
        )
        _tensors = [None] if not isinstance(tensors, list) else [None for i in tensors]

        for i in range(nb_layers):
            for j, x in enumerate(_tensors):
                if connection_type == "base":
                    _tensors[j] = _layers[i](
                        (tensors if not isinstance(tensors, list) else tensors[j])
                        if x is None
                        else x
                    )
                elif connection_type == "densely":
                    _densely[j].append(
                        _layers[i][0](
                            (tensors if not isinstance(tensors, list) else tensors[j])
                            if x is None
                            else x
                        )
                    )
                    _tensors[j] = _layers[i][1](_densely[j])
        return _tensors if len(_tensors) > 1 else _tensors[0]

    return _block


@ingredient.capture(prefix="layers")
def layer(
    conv1d: Dict = {},
    dropout: Dict = None,
    filters: Optional[int] = None,
    k: Optional[int] = None,
    densley: bool = False,
) -> Callable:
    """Get conv1d layer from config."""
    if filters is not None:
        conv1d = dict(conv1d, **{"filters": filters})
    elif k is not None:
        conv1d = dict(conv1d, **{"filters": k})
    if densley:
        conv1d = dict(conv1d, **{"padding": "same"})

    conv1d_layer = Conv1D.from_config(conv1d)
    if dropout and dropout["t"] == "layerwise":
        dropout_layer = deserialize_layer(dropout)

    def _layer(x: KerasTensor) -> KerasTensor:
        x = conv1d_layer(x)
        if dropout and dropout["t"] == "layerwise":
            x = dropout_layer(x)
        return x

    return _layer
