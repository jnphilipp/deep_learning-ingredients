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
"""Core module of callbacks ingredient."""

import os

from logging import Logger
from sacred import Ingredient
from sacred.run import Run
from sacred.observers import FileStorageObserver
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
    TensorBoard,
    TerminateOnNaN,
)
from typing import Any, Dict, List, Optional

from .print_sample_prediction import PrintSamplePrediction
from .sacred_metrics_logging import SacredMetricsLogging
from .weights_logging import WeightsLogging


ingredient = Ingredient("callbacks")


@ingredient.config
def _config():
    terminateonnan = True


@ingredient.capture
def get(
    _log: Logger,
    _run: Run,
    earlystopping: Optional[Dict[str, Any]] = None,
    modelcheckpoint: Optional[Dict[str, Any]] = None,
    learningratescheduler: Optional[Dict[str, Any]] = None,
    reducelronplateau: Optional[Dict[str, Any]] = None,
    tensorboard: Optional[Dict[str, Any]] = None,
    terminateonnan: bool = True,
    printsampleprediction: Optional[Dict[str, Any]] = None,
    weightslogging: Optional[Dict[str, str]] = None,
) -> List[Callback]:
    """Get callback list based on config."""
    _log.info("Add SacredMetricsLogging callback.")
    callbacks = [SacredMetricsLogging(_run)]

    if terminateonnan:
        _log.info("Add TerminateOnNaN callback.")
        callbacks.append(TerminateOnNaN())

    if weightslogging is not None:
        _log.info("Add WeightsLogging callback.")
        if len(_run.observers) >= 1 and type(_run.observers[0]) == FileStorageObserver:
            path = os.path.join(_run.observers[0].dir, "weights_history.csv")
            callbacks.append(
                WeightsLogging(
                    path=path,
                    **{k: v for k, v in weightslogging.items() if k != "path"}
                )
            )
        else:
            callbacks.append(WeightsLogging(**weightslogging))

    if modelcheckpoint is not None:
        _log.info("Add ModelCheckpoint callback.")

        if len(_run.observers) >= 1 and type(_run.observers[0]) == FileStorageObserver:
            base_dir = os.path.join(_run.observers[0].dir, "modelcheckpoint")
            os.makedirs(base_dir, exist_ok=True)
            filepath = os.path.join(base_dir, modelcheckpoint["filepath"])

            callbacks.append(
                ModelCheckpoint(
                    filepath,
                    **{k: v for k, v in modelcheckpoint.items() if k != "filepath"}
                )
            )
        else:
            callbacks.append(ModelCheckpoint(**modelcheckpoint))

    if learningratescheduler is not None:
        _log.info("Add LearningRateScheduler callback.")
        callbacks.append(LearningRateScheduler(**learningratescheduler))

    if earlystopping is not None:
        _log.info("Add EarlyStopping callback.")
        callbacks.append(EarlyStopping(**earlystopping))

    if reducelronplateau is not None:
        _log.info("Add ReduceLROnPlateau callback.")
        callbacks.append(ReduceLROnPlateau(**reducelronplateau))

    if printsampleprediction is not None:
        _log.info("Add PrintSamplePrediction callback.")
        callbacks.append(PrintSamplePrediction(**printsampleprediction))

    if tensorboard is not None:
        _log.info("Add TensorBoard callback.")

        if len(_run.observers) >= 1 and type(_run.observers[0]) == FileStorageObserver:
            log_dir = os.path.join(_run.observers[0].dir, "logs")
            os.makedirs(log_dir, exist_ok=True)

            callbacks.append(
                TensorBoard(
                    log_dir, **{k: v for k, v in tensorboard.items() if k != "log_dir"}
                )
            )
        else:
            callbacks.append(TensorBoard(**tensorboard))

    return callbacks
