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
from tensorflow.keras.layers import Activation, BatchNormalization, Dense
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Union

from . import ingredient
from .inputs import inputs
from .merge_layer import merge_layer
from .outputs import outputs


@ingredient.capture
def build(N: int, merge: Optional[Dict], layers: Dict, optimizer: Optimizer,
          _log: Logger, loss_weights: Optional[Union[List, Dict]] = None,
          sample_weight_mode: Optional[Union[str, Dict[str, str],
                                             List[str]]] = None,
          weighted_metrics: Optional[List] = None,
          target_tensors: Optional[Union[Tensor, List[Tensor]]] = None, *args,
          **kwargs) -> Model:
    if 'name' in kwargs:
        name = kwargs.pop('name')
    else:
        name = 'dense'
    _log.info(f'Build Dense model [{name}]')

    if 'units' in layers and type(layers['units']) == list:
        assert len(layers['units']) == N

    ins, xs = inputs(inputs=kwargs['inputs'], layers=layers) \
        if 'inputs' in kwargs else inputs()
    if merge is not None and 'depth' in merge and merge['depth'] == 0:
        xs = [merge_layer(xs, t=merge['t'],
                          config=merge['config'] if 'config' in merge else {})]

    tensors: dict = {}
    for i in range(N):
        for j, x in enumerate(xs):
            conf = dict(**layers['dense'])

            if 'output_shape' in kwargs and i == N - 1:
                if type(kwargs['output_shape']) == tuple:
                    conf['units'] = kwargs['output_shape'][0]
                else:
                    conf['units'] = kwargs['output_shape']
            elif 'units' in layers and type(layers['units']) == list:
                conf['units'] = layers['units'][i]

            if 'batchnorm' in layers:
                activation = conf['activation']
                conf['activation'] = 'linear'

            if i not in tensors and 'batchnorm' in layers:
                tensors[i] = (Dense.from_config(conf),
                              BatchNormalization.from_config(
                                layers['batchnorm']),
                              Activation(activation))
            elif i not in tensors:
                tensors[i] = Dense.from_config(conf)

            if 'batchnorm' in layers:
                xs[j] = tensors[i][2](tensors[i][1](tensors[i][0](x)))
            else:
                xs[j] = tensors[i](x)
            if 'dropout' in layers:
                xs[j] = deserialize_layer(layers['dropout'])(xs[j])

        if merge is not None and 'depth' in merge and merge['depth'] == i + 1:
            xs = [merge_layer(xs, t=merge['t'],
                              config=merge['config'] if 'config' in merge
                              else {})]

    # outputs
    outs, loss, metrics = outputs(xs, outputs=kwargs['outputs'],
                                  layers=layers) if 'outputs' in kwargs else \
        outputs(xs)

    # Model
    model = Model(inputs=ins, outputs=outs, name=name)
    model.compile(loss=loss, metrics=metrics, loss_weights=loss_weights,
                  optimizer=optimizer, sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
