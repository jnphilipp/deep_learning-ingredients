# -*- coding: utf-8 -*-

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
