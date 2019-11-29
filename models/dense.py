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

import math

from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from . import ingredient
from .outputs import outputs


@ingredient.capture
def build(input_shape, N, layers, optimizer, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          _log=None, *args, **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
        del kwargs['name']
        _log.info(f'Build Dense model [{name}]')
    else:
        name = 'dense'
        _log.info('Build Dense model')

    inputs = Input(input_shape, name='input')
    x = inputs

    if 'units' in layers and type(layers['units']) == list:
        assert len(layers['units']) == N

    for i in range(N):
        if 'units' in layers and type(layers['units']) == list:
            conf = dict(layers['dense_config'],
                        **{'units': layers['units'][i]})
        else:
            conf = layers['dense_config']

        if 'output_shape' in kwargs and i == N - 1:
            if type(kwargs['output_shape']) == tuple:
                conf['units'] = kwargs['output_shape'][0]
            else:
                conf['units'] = kwargs['output_shape']

        x = Dense.from_config(conf)(x)
        if 'dropout' in layers:
            x = deserialize_layer(layers['dropout'])(x)

    # outputs
    outs, loss, metrics = outputs(x)

    # Model
    model = Model(inputs=inputs, outputs=outs, name=name)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
