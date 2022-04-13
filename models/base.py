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
"""Base module for models ingredient."""

from tensorflow.keras.layers import (
    Add,
    Average,
    Concatenate,
    Dot,
    Flatten,
    Layer,
    Maximum,
    Minimum,
    Multiply,
    RepeatVector,
    Reshape,
    Subtract,
)
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Callable, Dict, Iterable, List, Union

from .ingredient import ingredient


@ingredient.capture(prefix="merge")
def merge(t: str, config: Dict = {}) -> Callable:
    """Get merge layer from config."""
    assert t in [
        "add",
        "average",
        "concatenate",
        "dot",
        "maximum",
        "minimum",
        "multiply",
        "subtract",
    ]

    _layer: Layer
    if t == "add":
        _layer = Add(**config)
    elif t == "average":
        _layer = Average(**config)
    elif t == "concatenate":
        _layer = Concatenate(**config)
    elif t == "dot":
        _layer = Dot(**config)
    elif t == "maximum":
        _layer = Maximum(**config)
    elif t == "minimum":
        _layer = Minimum(**config)
    elif t == "multiply":
        _layer = Multiply(**config)
    elif t == "subtract":
        _layer = Subtract(**config)

    def _merge(tensors: List[KerasTensor]) -> KerasTensor:
        return _layer(tensors)

    return _merge


@ingredient.capture(prefix="merge")
def zip_merge(t: str, config: Dict = {}) -> Callable:
    """Get merge layer doing a zip from config."""
    assert t in [
        "add",
        "average",
        "concatenate",
        "dot",
        "maximum",
        "minimum",
        "multiply",
        "subtract",
    ]

    _layer: Layer
    if t == "add":
        _layer = Add(**config)
    elif t == "average":
        _layer = Average(**config)
    elif t == "concatenate":
        _layer = Concatenate(**config)
    elif t == "dot":
        _layer = Dot(**config)
    elif t == "maximum":
        _layer = Maximum(**config)
    elif t == "minimum":
        _layer = Minimum(**config)
    elif t == "multiply":
        _layer = Multiply(**config)
    elif t == "subtract":
        _layer = Subtract(**config)

    def _zip_merge(tensors: List[List[KerasTensor]]) -> List[KerasTensor]:
        return [_layer(i) for i in zip(*tensors)]

    return _zip_merge


@ingredient.capture(prefix="reshape")
def reshape(target_shape: Iterable[int], **kwargs) -> Callable:
    """Get reshape layer from config."""
    reshape_layer = Reshape(target_shape, **kwargs)

    def _reshape(
        tensors: Union[KerasTensor, List[KerasTensor]]
    ) -> Union[KerasTensor, List[KerasTensor]]:
        xs = []
        for j, x in enumerate([tensors] if not isinstance(tensors, list) else tensors):
            xs.append(reshape_layer(x))
        return xs if len(xs) > 1 else xs[0]

    return _reshape


@ingredient.capture(prefix="flatten")
def flatten(**kwargs) -> Callable:
    """Get flatten layer from config."""
    flatten_layer = Flatten(**kwargs)

    def _flatten(
        tensors: Union[KerasTensor, List[KerasTensor]]
    ) -> Union[KerasTensor, List[KerasTensor]]:
        xs = []
        for j, x in enumerate([tensors] if not isinstance(tensors, list) else tensors):
            xs.append(flatten_layer(x))
        return xs if len(xs) > 1 else xs[0]

    return _flatten


@ingredient.capture(prefix="repeat_vector")
def repeat_vector(n: int, **kwargs) -> Callable:
    """Get repeat vector layer from config."""
    repeat_vector_layer = RepeatVector(n, **kwargs)

    def _repeat_vector(
        tensors: Union[KerasTensor, List[KerasTensor]]
    ) -> Union[KerasTensor, List[KerasTensor]]:
        xs = []
        for j, x in enumerate([tensors] if not isinstance(tensors, list) else tensors):
            xs.append(repeat_vector_layer(x))
        return xs if len(xs) > 1 else xs[0]

    return _repeat_vector
