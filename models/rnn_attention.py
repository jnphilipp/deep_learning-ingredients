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
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, TimeDistributed
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Union

from .ingredient import ingredient
from .inputs import inputs
from .merge_layer import merge_layer


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
        _log.info(f"Build RNN-Attention model [{name}]")
    else:
        name = "rnn"
        _log.info("Build RNN-Attention model")

    ins, xs = inputs(**kwargs["inputs"]) if "inputs" in kwargs else inputs()
    if merge is not None and "depth" in merge and merge["depth"] == 0:
        xs = [merge_layer(xs)]

    print(ins)
    print(xs)

    tensors: dict = {}
    outs = []
    for i in range(N):
        for j, (x_in, x_out) in enumerate(xs):
            # input
            rnn_layer = dict(**layers["recurrent"])
            rnn_layer["config"] = dict(
                rnn_layer["config"], **{"return_sequences": True, "return_state": True}
            )
            if (
                "bidirectional" in layers
                and layers["bidirectional"]
                and i not in tensors
            ):
                conf = dict(layers["bidirectional"], **{"layer": rnn_layer})
                tensors[i] = {"in": Bidirectional.from_config(conf)}
            elif i not in tensors:
                tensors[i] = {"in": deserialize_layer(rnn_layer)}
            xs[j][0], in_fwd_state, in_back_state = tensors[i]["in"](x_in)

            # output
            rnn_layer = dict(**layers["recurrent"])
            rnn_layer["config"] = dict(
                rnn_layer["config"], **{"return_sequences": True, "return_state": True}
            )
            rnn_layer["config"]["units"] *= 2
            tensors[i]["out"] = deserialize_layer(rnn_layer)
            xs[j][1], out_state = tensors[i]["out"](
                x_out, initial_state=Concatenate(axis=-1)([in_fwd_state, in_back_state])
            )

            x = deserialize_layer(**layers["attention"])([xs[j][0], xs[j][1]])
            x = Concatenate(axis=-1)([xs[j][1], x])
            outs.append(TimeDistributed(Dense(16, activation="softmax"))(x))

        if merge is not None and "depth" in merge and merge["depth"] == i + 1:
            xs = [merge_layer(xs)]

    # Model
    model = Model(inputs=ins, outputs=outs, name=name)
    model.compile(
        loss="categorical_crossentropy",
        metrics=["binary_accuracy", "precision", "recall"],
        loss_weights=loss_weights,
        optimizer=optimizer,
        sample_weight_mode=sample_weight_mode,
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
    )
    return model
