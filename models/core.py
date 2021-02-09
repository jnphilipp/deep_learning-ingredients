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

import os
import sys

from ingredients import optimizers
from logging import Logger
from sacred.run import Run
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.saved_model import SaveOptions
from typing import Callable, Dict, Optional, Sequence, Union

from . import autoencoder, cnn, dense, gan, rnn, rnn_attention, siamese
from .ingredient import ingredient


@ingredient.config
def config():
    path = None
    inputs = []
    layers = {}
    merge = {}
    outputs = []


@ingredient.capture
def get(
    path: Optional[str], net_type: str, _log: Logger, *args, **kwargs
) -> Union[Model, Sequence[Model]]:
    net_types = [
        "autoencoder",
        "rnn-attention",
        "cnn",
        "dense",
        "gan",
        "rnn",
        "siamese",
    ]
    assert net_type in net_types

    kwargs["optimizer"] = (
        optimizers.get(**kwargs["optimizer"])
        if "optimizer" in kwargs
        else optimizers.get()
    )
    if not path or not os.path.exists(path):
        if net_type == "autoencoder":
            model = autoencoder.build(*args, **kwargs)
        elif net_type == "cnn":
            model = cnn.build(*args, **kwargs)
        elif net_type == "dense":
            model = dense.build(*args, **kwargs)
        elif net_type == "gan":
            model = gan.build(*args, **kwargs)
        elif net_type == "rnn":
            model = rnn.build(*args, **kwargs)
        elif net_type == "rnn-attention":
            model = rnn_attention.build(*args, **kwargs)
        elif net_type == "siamese":
            model = siamese.build(*args, **kwargs)

        if "log_params" not in kwargs or kwargs["log_params"]:
            log_param_count(model)
        return model
    else:
        return load(path)


@ingredient.capture
def load(
    path: Union[str, Sequence[str]], _log: Logger
) -> Union[Model, Sequence[Model]]:
    if path is None:
        _log.critical("No path given to load model.")
    elif isinstance(path, str):
        _log.info(f"Load model [{path}]")
        return load_model(path)
    else:
        models = []
        for p in path:
            _log.info(f"Load model [{p}]")
            models.append(load_model(p))
            log_param_count(models)
        return models


@ingredient.capture
def log_param_count(model: Union[Model, Sequence[Model]], _log: Logger):
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
    _log.info("Print model summary.")
    model = get()
    model.summary()


@ingredient.command
def plot(_log: Logger, _run: Run):
    model = get()
    model.summary()
    _log.info(f"Plot {model.name}.")

    if len(_run.observers) == 0:
        path = f"{model.name}.png"
    else:
        path = os.path.join(_run.observers[0].run_dir, f"{model.name}.png")
    plot_model(model, to_file=path, show_shapes=True, expand_nested=True)
