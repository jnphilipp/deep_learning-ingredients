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

from logging import Logger
from keras.layers import Bidirectional
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.optimizers import Optimizer
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
