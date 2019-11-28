# -*- coding: utf-8 -*-

from logging import Logger
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from typing import Union

from . import ingredient
from .inputs import inputs
from .merge_layer import merge_layer
from .outputs import outputs


@ingredient.capture
def build(N: int, merge: dict, layers: dict, optimizer: Optimizer,
          _log: Logger, loss_weights: Union[list, dict] = None,
          sample_weight_mode: str = None, weighted_metrics: list = None,
          target_tensors=None, *args, **kwargs) -> Model:
    if 'name' in kwargs:
        name = kwargs.pop('name')
        _log.info(f'Build RNN model [{name}]')
    else:
        name = 'rnn'
        _log.info('Build RNN model')

    ins, xs = inputs()
    if 'depth' in merge and merge['depth'] == 0:
        xs = [merge_layer(xs)]

    tensors: dict = {}
    for i in range(N):
        for j, x in enumerate(xs):
            rnn_layer = dict(**layers['recurrent'])
            if i != N - 1:
                rnn_layer['config'] = dict(rnn_layer['config'],
                                           **{'return_sequences': True})

            if 'bidirectional' in layers and layers['bidirectional'] and \
                    i not in tensors:
                conf = dict(layers['bidirectional'], **{'layer': rnn_layer})
                tensors[i] = Bidirectional.from_config(conf)
            elif i not in tensors:
                tensors[j] = deserialize_layer(rnn_layer)
            xs[j] = tensors[i](x)

        if 'depth' in merge and merge['depth'] == i + 1:
            xs = [merge_layer(xs)]

    # outputs
    outs, loss, metrics = outputs(xs)

    # Model
    model = Model(inputs=ins, outputs=outs, name=name)
    model.compile(loss=loss, metrics=metrics, loss_weights=loss_weights,
                  optimizer=optimizer, sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
