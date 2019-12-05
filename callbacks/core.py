# -*- coding: utf-8 -*-
# Copyright (C) 2019 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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

from logging import Logger
from sacred.run import Run
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TerminateOnNaN)
from typing import Any, Dict, List

from . import ingredient, PrintSamplePrediction, WeightsLogging


@ingredient.config
def config():
    terminateonnan = True


@ingredient.capture
def get(_log: Logger, _run: Run, earlystopping: Dict[str, Any] = None,
        modelcheckpoint: Dict[str, Any] = None, terminateonnan: bool = True,
        reducelronplateau: Dict[str, Any] = None,
        printsampleprediction: Dict[str, Any] = None,
        weightslogging: Dict[str, str] = None) -> List[Callback]:
    callbacks = []

    if terminateonnan:
        _log.info('Add TerminateOnNaN callback.')
        callbacks.append(TerminateOnNaN())

    if weightslogging is not None:
        _log.info('Add WeightsLogging callback.')
        path = os.path.join(_run.observers[0].run_dir, 'weights_history.csv')
        callbacks.append(WeightsLogging(path=path, **weightslogging))

    if modelcheckpoint is not None:
        _log.info('Add ModelCheckpoint callback.')

        base_dir = os.path.join(_run.observers[0].run_dir, 'modelcheckpoint')
        os.makedirs(base_dir, exist_ok=True)

        filepath = os.path.join(base_dir, modelcheckpoint['filepath'])
        kwargs = {k: v for k, v in modelcheckpoint.items() if k != 'filepath'}
        callbacks.append(ModelCheckpoint(filepath, **kwargs))

    if earlystopping is not None:
        _log.info('Add EarlyStopping callback.')
        callbacks.append(EarlyStopping(**earlystopping))

    if reducelronplateau is not None:
        _log.info('Add ReduceLROnPlateau callback.')
        callbacks.append(ReduceLROnPlateau(**reducelronplateau))

    if printsampleprediction is not None:
        _log.info('Add PrintSamplePrediction callback.')
        callbacks.append(PrintSamplePrediction(**printsampleprediction))

    return callbacks
