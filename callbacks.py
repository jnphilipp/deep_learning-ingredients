# -*- coding: utf-8 -*-

import os

from sacred import Ingredient

from keras.callbacks import *


ingredient = Ingredient('callbacks')


@ingredient.config
def config():
    terminateonnan = True


@ingredient.capture
def get(earlystopping=None, modelcheckpoint=None, reducelronplateau=None,
        terminateonnan=True, _log=None, _run=None):
    callbacks = []

    if terminateonnan:
        _log.info('Add TerminateOnNaN callback.')
        callbacks.append(TerminateOnNaN())

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

    return callbacks
