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

from logging import Logger
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Union

from .core import ingredient
from .inputs import inputs
from .merge_layer import merge_layer
from .outputs import outputs


@ingredient.capture
def build(
    N: int,
    merge: Optional[Dict],
    layers: Dict,
    optimizer: Optimizer,
    _log: Logger,
    loss_weights: Optional[Union[List, Dict]] = None,
    sample_weight_mode: Optional[Union[str, Dict[str, str], List[str]]] = None,
    weighted_metrics: Optional[List] = None,
    target_tensors: Optional[Union[Tensor, List[Tensor]]] = None,
    *args,
    **kwargs,
) -> Model:
    if "name" in kwargs:
        name = kwargs.pop("name")
    else:
        name = "rnn"
    _log.info(f"Build RNN model [{name}]")

    ins, xs = (
        inputs(inputs=kwargs["inputs"], layers=layers)
        if "inputs" in kwargs
        else inputs()
    )
    if merge is not None and "depth" in merge and merge["depth"] == 0:
        xs = [
            merge_layer(
                xs, t=merge["t"], config=merge["config"] if "config" in merge else {}
            )
        ]

    tensors: dict = {}
    for i in range(N):
        for j, x in enumerate(xs):
            rnn_layer = dict(**layers["recurrent"])
            if i != N - 1:
                rnn_layer["config"] = dict(
                    rnn_layer["config"], **{"return_sequences": True}
                )

            if (
                "bidirectional" in layers
                and layers["bidirectional"]
                and i not in tensors
            ):
                conf = dict(layers["bidirectional"], **{"layer": rnn_layer})
                tensors[i] = Bidirectional.from_config(conf)
            elif i not in tensors:
                tensors[i] = deserialize_layer(rnn_layer)
            xs[j] = tensors[i](x)

        if merge is not None and "depth" in merge and merge["depth"] == i + 1:
            xs = [
                merge_layer(
                    xs,
                    t=merge["t"],
                    config=merge["config"] if "config" in merge else {},
                )
            ]

    # outputs
    outs, loss, metrics = (
        outputs(xs, outputs=kwargs["outputs"], layers=layers)
        if "outputs" in kwargs
        else outputs(xs)
    )

    # Model
    model = Model(inputs=ins, outputs=outs, name=name)
    model.compile(
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
        optimizer=optimizer,
        sample_weight_mode=sample_weight_mode,
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
    )
    return model
