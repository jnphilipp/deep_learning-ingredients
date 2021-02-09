# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020
#               J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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
from tensorflow.keras import backend as B
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Union

from . import core
from .ingredient import ingredient


@ingredient.capture
def build(
    inner_net_type: str,
    outputs: Dict,
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
        name = "siamese"
    _log.info(f"Build Siamese [{inner_net_type}] model [{name}]")

    inner_model = core.get(
        None, inner_net_type, outputs=[{"t": "vec", "loss": "mse"}], *args, **kwargs
    )

    input_r = Input(inner_model.get_layer("input").input_shape[1:], name="input_r")
    input_l = Input(inner_model.get_layer("input").input_shape[1:], name="input_l")

    xr = inner_model(input_r)
    xl = inner_model(input_l)

    # outputs
    output_types = ["distance"]
    assert set([o["t"] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = {}
    for output in outputs:
        loss.append(output["loss"])
        if "metrics" in output:
            metrics[output["name"]] = output["metrics"]

        if output["t"] == "distance":
            outs.append(
                Lambda(
                    lambda x: B.mean(B.abs(x[0] - x[1]), axis=-1),
                    name=output["name"],
                    output_shape=(1,),
                )([xr, xl])
            )

    siamese_model = Model(inputs=[input_r, input_l], outputs=outs, name=name)
    siamese_model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        loss_weights=loss_weights,
        sample_weight_mode=sample_weight_mode,
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
    )
    return inner_model, siamese_model
