# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import deserialize as deserialize_layer, Input, Dense
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.models import ingredients


@ingredients.capture
def build(input_shape, N, layers, outputs, optimizer, _log, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          *args, **kwargs):
    if 'name' in kwargs:
        _log.info('Build Dense model [%s]' % kwargs['name'])
    else:
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

        x = Dense.from_config(conf)(x)
        if 'dropout' in layers:
            x = deserialize_layer(layers['dropout'])(x)

    # outputs
    assert set([o['t'] for o in outputs]).issubset(['class', 'vec'])

    outs = []
    loss = []
    metrics = []
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics.append(output['metrics'])

        if output['t'] == 'class':
            conf = dict(layers['dense_config'],
                        **{'units': output['nb_classes'],
                           'activation': output['activation']})
            if 'name' in output:
                conf['name'] = output['name']
            outs.append(Dense.from_config(conf)(x))
        elif output['t'] == 'vec':
            outs.append(x)

    # Model
    model = Model(inputs=inputs, outputs=outs,
                  name=kwargs['name'] if 'name' in kwargs else 'dense')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
