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
"""Conv2d module for models ingredient."""

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Flatten,
    Layer,
    Multiply,
    Reshape,
)
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Callable, Dict, List, Optional, Tuple, Union

from .ingredient import ingredient


@ingredient.capture(prefix="layers")
def block(
    nb_layers: int,
    connection_type: str = "base",
    do_pooling: bool = True,
    attention2d: Optional[Dict] = None,
    batchnorm: Optional[Dict] = None,
    bottleneck2d: Optional[Dict] = None,
    conv2d: Dict = {},
    pooling: Dict = {},
    concat_axis: int = -1,
    dropout: Optional[Dict] = None,
    theta: Optional[float] = None,
    filters: Optional[int] = None,
    k: Optional[int] = None,
    shortcuts: bool = True,
    **kwargs,
) -> Callable:
    """Get conv2d block from config."""
    assert connection_type in ["base", "densely"]
    if connection_type == "densely":
        assert concat_axis is not None
    nb_filters = kwargs["nb_filters"] if "nb_filters" in kwargs else 0
    pooling = pooling.copy()

    _layers: Dict[int, Layer] = {}
    for i in range(nb_layers):
        if k is not None:
            nb_filters += k

        if connection_type == "base":
            _layers[i] = layer(filters=filters)
        elif connection_type == "densely":
            _layers[i] = (layer(), Concatenate(axis=concat_axis))

    shortcuts_nb_filters = nb_filters

    if attention2d is not None:
        attention2d_layer = Conv2D.from_config(dict(attention2d, **{"filters": 1}))
        attention2d_flatten_layer = Flatten()
        attention2d_activation_layer = Activation()
        attention2d_reshape_layer = Reshape(attention2d_layer.output_shape[1:])
        attention2d_multiply_layer = Multiply()

    if do_pooling:
        if theta is not None:
            nb_filters = int(nb_filters * theta)

        if batchnorm is not None:
            if bottleneck2d is not None:
                bottleneck_activation = bottleneck2d["activation"]
                bottleneck2d = dict(bottleneck2d, **{"activation": "linear"})

            if pooling["class_name"].lower() == "conv2d":
                pooling_activation = conv2d["activation"]
                pooling["config"] = dict(pooling["config"], **{"activation": "linear"})

        if bottleneck2d is not None:
            assert k is not None

            if theta is not None:
                bottleneck2d = dict(bottleneck2d, **{"filters": nb_filters})
            else:
                factor = bottleneck2d["filters"]
                bottleneck2d = dict(bottleneck2d, **{"filters": factor * k})
            pooling_bottleneck2d_layer = Conv2D.from_config(bottleneck2d)
            if batchnorm is not None:
                pooling_bottleneck2d_bn_layer = BatchNormalization.from_config(
                    batchnorm
                )
                pooling_bottleneck2d_activation_layer = Activation(
                    bottleneck_activation
                )

        if filters is not None:
            pooling["config"] = dict(pooling["config"], **{"filters": filters})
        elif k is not None:
            if theta is not None:
                pooling["config"] = dict(pooling["config"], **{"filters": nb_filters})
            else:
                pooling["config"] = dict(pooling["config"], **{"filters": k})

        pooling_layer = deserialize_layer(pooling)
        if batchnorm is not None and pooling["class_name"].lower() == "conv2d":
            pooling_bn_layer = BatchNormalization.from_config(batchnorm)
            pooling_activation_layer = Activation(pooling_activation)
        if dropout and dropout["t"] in ["blockwise", "layerwise"]:
            dropout_layer = deserialize_layer(dropout)

    def _block(
        tensors: Union[KerasTensor, List[KerasTensor]]
    ) -> Union[
        Union[KerasTensor, List[KerasTensor]],
        Tuple[
            Union[KerasTensor, List[KerasTensor]],
            Union[Tuple[KerasTensor, int], List[Tuple[KerasTensor, int]]],
        ],
    ]:
        _densely = (
            [[tensors]] if not isinstance(tensors, list) else [[i] for i in tensors]
        )
        _tensors = [None] if not isinstance(tensors, list) else [None for i in tensors]
        _shortcuts = []

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

        for j, x in enumerate(_tensors):
            _shortcuts.append((_tensors[j], shortcuts_nb_filters))

            if attention2d_layer:
                w = attention2d_layer(_tensors[j])
                w_shape = w._shape_val[1:]
                w = attention2d_flatten_layer(w)
                w = attention2d_activation_layer(w)
                w = attention2d_reshape_layer(w_shape)(w)
                _tensors[j] = attention2d_multiply_layer(_tensors[j], w)

            if pooling_bottleneck2d_layer:
                _tensors[j] = pooling_bottleneck2d_layer(_tensors[j])
                if pooling_bottleneck2d_bn_layer:
                    _tensors[j] = pooling_bottleneck2d_bn_layer(_tensors[j])
                    _tensors[j] = pooling_bottleneck2d_activation_layer(_tensors[j])

            if pooling_layer:
                _tensors[j] = pooling_layer(_tensors[j])
                if pooling_bn_layer:
                    _tensors[j] = pooling_bn_layer(_tensors[j])
                    _tensors[j] = pooling_activation_layer(_tensors[j])
                if dropout_layer:
                    _tensors[j] = dropout_layer(_tensors[j])

        if shortcuts:
            return (
                _tensors if len(_tensors) > 1 else _tensors[0],
                _shortcuts if len(_tensors) > 1 else _shortcuts[0],
            )
        else:
            return _tensors if len(_tensors) > 1 else _tensors[0]

    return _block


@ingredient.capture(prefix="layers")
def layer(
    batchnorm: Optional[Dict] = None,
    bottleneck2d: Optional[Dict] = None,
    conv2d: Dict = {},
    dropout: Optional[Dict] = None,
    filters: Optional[int] = None,
    k: Optional[int] = None,
) -> Callable:
    """Get conv2d layer from cconfig."""
    if batchnorm is not None:
        if bottleneck2d is not None:
            bottleneck_activation = bottleneck2d["activation"]
            bottleneck2d = dict(bottleneck2d, **{"activation": "linear"})

        conv_activation = conv2d["activation"]
        conv2d = dict(conv2d, **{"activation": "linear"})

    if bottleneck2d is not None:
        assert k is not None

        factor = bottleneck2d["filters"]
        bottleneck2d_layer = Conv2D.from_config(
            dict(bottleneck2d, **{"filters": factor * k})
        )
        if batchnorm is not None:
            bottleneck2d_bn_layer = BatchNormalization.from_config(batchnorm)
            bottleneck2d_activation_layer = Activation(bottleneck_activation)

    if filters is not None:
        conv2d = dict(conv2d, **{"filters": filters})
    elif k is not None:
        conv2d = dict(conv2d, **{"filters": k})

    conv2d_layer = Conv2D.from_config(conv2d)
    if batchnorm is not None:
        conv2d_bn_layer = BatchNormalization.from_config(batchnorm)
        conv2d_activation_layer = Activation(conv_activation)
    if dropout and dropout["t"] == "layerwise":
        dropout_layer = deserialize_layer(dropout)

    def _layer(x: KerasTensor) -> KerasTensor:
        if bottleneck2d_layer:
            x = bottleneck2d_layer(x)
            if bottleneck2d_bn_layer:
                x = bottleneck2d_bn_layer(x)
                x = bottleneck2d_activation_layer(x)
        x = conv2d_layer(x)
        if conv2d_bn_layer:
            x = conv2d_bn_layer(x)
            x = conv2d_activation_layer(x)
        if dropout_layer:
            x = dropout_layer(x)
        return x

    return _layer
