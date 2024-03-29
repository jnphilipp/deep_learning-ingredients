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
"""Outputs module for models ingredient."""

from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    RepeatVector,
)
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from typing import Dict, List, Optional, Tuple, Union

from . import metrics, losses
from .base import merge
from .ingredient import ingredient


@ingredient.capture
def build(
    vecs: Union[KerasTensor, List[KerasTensor]],
    layers: Dict,
    outputs: List[Dict],
    *args,
    **kwargs,
) -> Tuple[List[KerasTensor], Dict[str, Loss], Dict[str, List[Metric]]]:
    """Build model output(s) from config."""
    output_types = ["class", "image", "mask", "seq", "vec"]
    assert set([o["t"] for o in outputs]).issubset(output_types)

    if "batchnorm" in layers:
        batchnorm = layers["batchnorm"]
        if "bottleneck2d" in layers:
            bottleneck_activation = layers["bottleneck2d"]["activation"]
            bottleneck2d: Optional[dict] = dict(
                layers["bottleneck2d"], **{"activation": "linear"}
            )
        else:
            bottleneck2d = None
    else:
        batchnorm = None
        if "bottleneck2d" in layers:
            bottleneck2d = layers["bottleneck2d"]
        else:
            bottleneck2d = None
    concat_axis = layers["concat_axis"] if "concat_axis" in layers else -1
    conv1d = layers["conv1d"] if "conv1d" in layers else {}
    conv2d = layers["conv2d"] if "conv2d" in layers else {}
    conv2dt = layers["conv2dt"] if "conv2dt" in layers else {}
    dense = layers["dense"] if "dense" in layers else {}
    rpvec = layers["repeatvector"] if "repeatvector" in layers else {}

    if not isinstance(vecs, list):
        vecs = [vecs]

    if len(outputs) > 1 and len(vecs) == 1:
        vecs = vecs * len(outputs)
    elif len(outputs) == 1 and len(vecs) > 1:
        vecs = [merge()(vecs)]
    else:
        assert len(outputs) == len(vecs)

    outs = []
    loss = {}
    metrics_dict: Dict[str, List[Metric]] = {}
    for v, output in zip(vecs, outputs):
        activation = output["activation"]
        name = f'output_{output["name"]}'
        nb_classes = output["nb_classes"] if "nb_classes" in output else 1

        loss[name] = losses.get(output["loss"])
        if "metrics" in output:
            metrics_dict[name] = metrics.get(output["metrics"])

        if output["t"] == "class":
            if output["layer"] == "conv1d":
                outs.append(
                    Conv1D.from_config(
                        dict(
                            conv1d,
                            **{
                                "filters": nb_classes,
                                "activation": activation,
                                "name": name,
                            },
                        )
                    )(v)
                )
            elif output["layer"] == "conv2d":
                x = Conv2D.from_config(
                    dict(
                        conv2d,
                        **{
                            "filters": nb_classes,
                            "kernel_size": v._shape_val[1:3],
                            "padding": "valid",
                        },
                    )
                )(v)
                x = Flatten()(x)
                outs.append(Activation(activation, name=name)(x))
            elif output["layer"] == "dense":
                outs.append(
                    Dense.from_config(
                        dict(
                            dense,
                            **{
                                "units": nb_classes,
                                "activation": activation,
                                "name": name,
                            },
                        )
                    )(v)
                )
        elif output["t"] == "image":
            outs.append(
                Conv2D.from_config(
                    dict(
                        conv2d,
                        **{
                            "filters": 1 if output["grayscale"] else 3,
                            "kernel_size": (1, 1),
                            "padding": "same",
                            "activation": activation,
                            "name": name,
                        },
                    )
                )(v)
            )
        elif output["t"] == "mask":
            shortcuts = kwargs["shortcuts"]

            s = v
            for i in reversed(range(len(shortcuts))):
                shortcut = shortcuts[i][0]
                filters = shortcuts[i - 1 if i >= 0 else 0][1]
                if i is not len(shortcuts) - 1:
                    s = concatenate([s, shortcut], axis=concat_axis)
                else:
                    s = shortcut
                if i > 0:
                    if bottleneck2d is not None:
                        s = Conv2D.from_config(
                            dict(bottleneck2d, **{"filters": filters})
                        )(s)
                        if batchnorm is not None:
                            s = BatchNormalization.from_config(batchnorm)(s)
                            s = Activation(bottleneck_activation)(s)
                    s = Conv2DTranspose.from_config(
                        dict(conv2dt, **{"filters": filters})
                    )(s)

            outs.append(
                Conv2D.from_config(
                    dict(
                        conv2d,
                        **{
                            "filters": nb_classes,
                            "name": name,
                            "activation": activation,
                        },
                    )
                )(s)
            )
        elif output["t"] == "seq":
            x = RepeatVector.from_config(dict(rpvec, **{"n": output["length"]}))(v)

            for j in range(output["N"] if "N" in output else 1):
                if "recurrent" in output:
                    rnn_layer = dict(**layers[output["recurrent"]])
                elif "bidirectional" in output:
                    rnn_layer = dict(**layers[output["bidirectional"]])
                elif "recurrent_out" in output:
                    rnn_layer = dict(**layers[output["recurrent_out"]])
                elif "recurrent_out" in layers:
                    rnn_layer = dict(**layers["recurrent_out"])
                elif "recurrent" in layers:
                    rnn_layer = dict(**layers["recurrent"])
                rnn_layer["config"] = dict(
                    rnn_layer["config"], **{"return_sequences": True}
                )

                if "bidirectional" in output:
                    conf = dict(layers["bidirectional"], **{"layer": rnn_layer})
                    x = Bidirectional.from_config(conf)(x)
                else:
                    x = deserialize_layer(rnn_layer)(x)
            outs.append(
                Conv1D.from_config(
                    dict(
                        conv1d,
                        **{
                            "filters": nb_classes,
                            "activation": activation,
                            "name": name,
                        },
                    )
                )(x)
            )
        elif output["t"] == "vec":
            outs.append(v)

    return outs, loss, metrics_dict
