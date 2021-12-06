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
"""Core module of models ingredient."""

import os
import sys

from ingredients import optimizers
from logging import Logger
from sacred.run import Run
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import plot_model
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.saved_model import SaveOptions
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from . import base, conv1d, conv2d, dense, inputs, outputs, rnn
from .ingredient import ingredient


@ingredient.config
def _config():
    path = None
    inputs = []
    layers = {}
    merge = {}
    outputs = []


@ingredient.capture
def get(
    path: Optional[str], _log: Logger, *args, **kwargs
) -> Union[Model, Sequence[Model]]:
    """Build or load model(s)."""
    if not path or not os.path.exists(path):
        kwargs["optimizer"] = (
            optimizers.get(**kwargs["optimizer"])
            if "optimizer" in kwargs
            else optimizers.get()
        )

        model = build(*args, **kwargs)

        if "log_params" not in kwargs or kwargs["log_params"]:
            log_param_count(model)
        return model
    else:
        return load(path)


@ingredient.capture
def build(
    blocks: List[Dict],
    optimizer: Optimizer,
    _log: Logger,
    loss_weights: Optional[Union[List, Dict]] = None,
    sample_weight_mode: Optional[Union[str, Dict[str, str], List[str]]] = None,
    weighted_metrics: Optional[List] = None,
    target_tensors: Optional[
        Union[Tensor, KerasTensor, List[Tensor], List[KerasTensor]]
    ] = None,
    *args,
    **kwargs,
) -> Model:
    """Build model from config."""
    if "name" in kwargs:
        name = kwargs.pop("name")
        _log.info(f"Build model [{name}].")
    else:
        name = None
        _log.info("Build model.")

    # inputs
    model_inputs, input_tensors = inputs.build()

    tensors: List[KerasTensor] = []
    shortcuts = []
    output_tensors = []
    for i, block in enumerate(blocks):
        if "inputs" in block:
            if isinstance(block["inputs"], int) and block["inputs"] >= 0:
                x = tensors[block["inputs"]]
            elif isinstance(block["inputs"], str) and block["inputs"] == "inputs":
                x = input_tensors
            elif isinstance(block["inputs"], Iterable):
                x = []
                for j in block["inputs"]:
                    if isinstance(j, int):
                        x.append(tensors[j])
                    elif isinstance(j, str):
                        if j == "inputs":
                            x.append(input_tensors)
        elif len(tensors) == 0:
            x = input_tensors
        else:
            x = tensors[-1]

        if block["t"] == "conv1d":
            tensors.append(
                conv1d.block(**block["config"] if "config" in block else {})(x)
            )
        elif block["t"] == "conv2d":
            t = conv2d.block(**block["config"] if "config" in block else {})(x)
            if isinstance(t, tuple):
                tensors.append(t[0])
                shortcuts.append(t[1])
            else:
                tensors.append(t)
        elif block["t"] == "dense":
            tensors.append(
                dense.block(**block["config"] if "config" in block else {})(x)
            )
        elif block["t"] == "flatten":
            tensors.append(
                base.flatten(**block["config"] if "config" in block else {})(x)
            )
        elif block["t"] == "merge":
            tensors.append(
                base.merge(**block["config"] if "config" in block else {})(x)
            )
        elif block["t"] == "reshape":
            tensors.append(
                base.reshape(**block["config"] if "config" in block else {})(x)
            )
        elif block["t"] == "rnn":
            tensors.append(rnn.block(**block["config"] if "config" in block else {})(x))
        elif block["t"] == "zip":
            tensors.append(
                base.zip_merge(**block["config"] if "config" in block else {})(x)
            )

        if "output" in block and block["ouput"] is True:
            if isinstance(tensors[-1], list):
                output_tensors += tensors[-1]
            else:
                output_tensors.append(tensors[-1])

    if len(output_tensors) == 0:
        if isinstance(tensors[-1], list):
            output_tensors += tensors[-1]
        else:
            output_tensors.append(tensors[-1])

    # outputs
    model_outputs, loss, metrics = outputs.build(output_tensors, shortcuts=shortcuts)

    # Model
    model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        loss_weights=loss_weights,
        sample_weight_mode=sample_weight_mode,
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
    )
    return model


@ingredient.capture
def load(
    path: Union[str, Sequence[str]], _log: Logger
) -> Union[Model, Sequence[Model]]:
    """Load model(s) from path(s)."""
    if path is None:
        _log.critical("No path given to load model.")
    elif isinstance(path, str):
        _log.info(f"Load model [{path}]")
        model = load_model(path)
        log_param_count(model)
        return model
    else:
        models = []
        for p in path:
            _log.info(f"Load model [{p}]")
            models.append(load_model(p))
            log_param_count(models)
        return models


@ingredient.capture
def log_param_count(model: Union[Model, Sequence[Model]], _log: Logger):
    """Log param count."""
    if isinstance(model, Model):
        model = [model]
    for m in model:
        if hasattr(model, "_collected_trainable_weights"):
            m._check_trainable_weights_consistency()
            trainable_count = count_params(m._collected_trainable_weights)
        else:
            trainable_count = count_params(m.trainable_weights)
        non_trainable_count = count_params(m.non_trainable_weights)

        _log.info(f"Model: {m.name}")
        _log.info(f"  Total params: {trainable_count + non_trainable_count:,}")
        _log.info(f"  Trainable params: {trainable_count:,}")
        _log.info(f"  Non-trainable params: {non_trainable_count:,}")


@ingredient.capture
def save(
    model: Model,
    path: str,
    _log: Logger,
    overwrite: bool = True,
    include_optimizer: bool = True,
    save_format: Optional[str] = None,
    signatures: Optional[Union[Callable, Dict[str, Callable]]] = None,
    options: SaveOptions = None,
    **kwargs,
):
    """Save model(s)."""
    if "name" in kwargs:
        name = kwargs.pop("name")
    else:
        name = "model"

    _log.info(f"Save model [{name}]")
    with open(os.path.join(path, f"{name}.json"), "w", encoding="utf8") as f:
        f.write(model.to_json())
        f.write("\n")

    stdout = sys.stdout
    with open(os.path.join(path, f"{name}.summary"), "w", encoding="utf8") as f:
        sys.stdout = f
        model.summary()
    sys.stdout = stdout

    model.save(
        os.path.join(path, name),
        overwrite,
        include_optimizer,
        save_format,
        signatures,
        options,
    )


@ingredient.command
def summary(_log: Logger):
    """Model(s) summary command."""
    _log.info("Print model summary.")
    model = get()
    model.summary()


@ingredient.command
def to_json(_log: Logger):
    """Model(s) to_json command."""
    _log.info("Print model.to_json().")
    model = get()
    _log.info(model.to_json())


@ingredient.command
def plot(_log: Logger, _run: Run):
    """Plot model(s) command."""
    model = get()
    model.summary()
    _log.info(f"Plot {model.name}.")

    if len(_run.observers) == 0:
        path = f"{model.name}.png"
    else:
        path = os.path.join(_run.observers[0].run_dir, f"{model.name}.png")
    plot_model(model, to_file=path, show_shapes=True, expand_nested=True)
