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
"""Inputs module of callbacks ingredient."""

from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Dict, List, Tuple

from .base import merge
from .ingredient import ingredient


@ingredient.capture
def build(
    inputs: List[Dict], layers: Dict, *args, **kwargs
) -> Tuple[List[KerasTensor], List[KerasTensor]]:
    """Transform config to model inputs."""
    model_inputs = []
    xs = []
    tensors: dict = kwargs["tensors"] if "tensors" in kwargs else {}
    for _input in inputs:
        if _input["t"] == "embedding" or _input["t"] == "embedding-attention":
            x = Input(shape=_input["shape"], name=f'input_{_input["name"]}')
            if _input["t"] == "embedding-attention":
                o = Input(shape=_input["shape"], name=f'input_output_{_input["name"]}')
                model_inputs.append([x, o])
            else:
                model_inputs.append(x)

            if "embedding" not in tensors:
                tensors["embedding"] = Embedding.from_config(layers["embedding"])

            x = tensors["embedding"](x)
            if _input["t"] == "embedding-attention":
                o = tensors["embedding"](o)

            if layers["embedding_dropout"]:
                if layers["embedding_dropout"]["class_name"] not in tensors:
                    tensors[
                        layers["embedding_dropout"]["class_name"]
                    ] = deserialize_layer(layers["embedding_dropout"])

                x = tensors[layers["embedding_dropout"]["class_name"]](x)
                if _input["t"] == "embedding-attention":
                    o = tensors[layers["embedding_dropout"]["class_name"]](o)

            if _input["t"] == "embedding-attention":
                xs.append([x, o])
            else:
                xs.append(x)
        elif _input["t"] == "noise":
            x = Input(shape=_input["shape"], name=f'input_{_input["name"]}')
            model_inputs.append(x)
            xs.append(deserialize_layer(layers["noise"])(x))
        elif _input["t"] == "input" or _input["t"] == "input-attention":
            x = Input(shape=_input["shape"], name=f'input_{_input["name"]}')
            model_inputs.append(x)
            xs.append(x)

            if _input["t"] == "embedding-attention":
                x = Input(shape=_input["shape"], name=f'input_output_{_input["name"]}')
                model_inputs.append(x)
                xs.append(x)
        elif _input["t"] == "merge":
            _ins, _xs = build(_input["inputs"], tensors=tensors, *args, **kwargs)
            model_inputs += _ins

            xs += [
                merge(
                    t=_input["merge"]["t"],
                    config=_input["merge"]["config"]
                    if "config" in _input["merge"]
                    else {},
                )(_xs)
            ]

    return model_inputs, xs
