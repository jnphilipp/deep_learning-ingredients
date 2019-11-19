# -*- coding: utf-8 -*-

from logging import Logger
from tensorflow.keras.layers import Bidirectional, Conv1D, RepeatVector
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from typing import Iterable, Union

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
        name = kwargs['name']
        del kwargs['name']
        _log.info(f'Build Seq2Seq model [{name}]')
    else:
        name = 'seq2seq'
        _log.info('Build Seq2Seq model')

    ins, xs = inputs()
    if ('depth' not in merge and len(xs) > 1) or \
            ('depth' in merge and merge['depth'] == 0):
        if 't' not in merge:
            xs = [merge_layer(xs, t='concatenate')]
        else:
            xs = [merge_layer(xs)]

    tensors: dict = {}
    for j in range(N):
        for i in range(len(xs)):
            x = xs[i]

            if 'recurrent_in' in layers:
                rnn_layer = dict(**layers['recurrent_in'])
            else:
                rnn_layer = dict(**layers['recurrent'])
            if j != N - 1:
                rnn_layer['config'] = dict(rnn_layer['config'],
                                           **{'return_sequences': True})

            if 'bidirectional' in layers and layers['bidirectional'] and \
                    j not in tensors:
                conf = dict(layers['bidirectional'], **{'layer': rnn_layer})
                tensors[j] = Bidirectional.from_config(conf)
            elif j not in tensors:
                tensors[j] = deserialize_layer(rnn_layer)
            xs[i] = tensors[j](x)

        if 'depth' in merge and merge['depth'] == j + 1:
            xs = [merge_layer(xs)]

    # outputs
    outs, loss, metrics = outputs(xs)

    # Model
    model = Model(inputs=ins, outputs=outs, name=name)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
