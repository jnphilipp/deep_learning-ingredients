# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Activation, Input, Lambda
from keras.models import Model
from keras.optimizers import deserialize
from ingredients import models
from ingredients.models import ingredients


@ingredients.capture
def build(inner_net_type, outputs, optimizer, _log, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          *args, **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
        del kwargs['name']
        _log.info('Build Siamese [%s] model [%s]' % (inner_net_type, name))
    else:
        name = None
        _log.info('Build Siamese [%s] model' % inner_net_type)

    inner_model = models.get(None, inner_net_type,
                             outputs=[{'t': 'vec', 'loss': 'mse'}], *args,
                             **kwargs)

    input_r = Input(inner_model.get_layer('input').input_shape[1:],
                    name='input_r')
    input_l = Input(inner_model.get_layer('input').input_shape[1:],
                    name='input_l')

    xr = inner_model(input_r)
    xl = inner_model(input_l)

    # outputs
    output_types = ['distance']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = []
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics.append(output['metrics'])

        if output['t'] == 'distance':
            outs.append(Lambda(lambda x: K.mean(K.abs(x[0] - x[1]), axis=-1),
                               name=output['name'],
                               output_shape=(1,))([xr, xl]))

    siamese_model = Model(inputs=[input_r, input_l], outputs=outs,
                          name=name if name else 'siamese')
    siamese_model.compile(loss=loss, optimizer=deserialize(optimizer),
                          metrics=metrics, loss_weights=loss_weights,
                          sample_weight_mode=sample_weight_mode,
                          weighted_metrics=weighted_metrics,
                          target_tensors=target_tensors)
    return siamese_model, inner_model
