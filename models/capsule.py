# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import *
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers
from keras_contrib.layers import Capsule

from . import ingredient
from .. import models


@ingredient.capture
def build(input_model_type, layers, loss, metrics, optimizer, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          _log=None, *args, **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
        _log.info('Build Capsule model [%s]' % kwargs['name'])
    else:
        name = 'capsule'
        _log.info('Build Capsule model')

    input_model = models.get(None, input_model_type,
                             outputs=[{'t': 'vec', 'loss': 'mse'}],
                             callbacks=False, *args, **kwargs)
    if K.image_data_format() == 'channels_first':
        filters = int(input_model.outputs[0].shape[1])
    else:
        filters = int(input_model.outputs[0].shape[-1])

    inputs = Input(shape=input_model.get_layer('input').input_shape[1:],
                   name='input')
    x = input_model(inputs)
    x = Reshape((-1, filters))(x)
    x = Capsule.from_config(layers['capsule_config'])(x)
    outputs = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(x)

    # Model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics, loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
