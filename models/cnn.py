# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import *
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers

from . import ingredient
from .outputs import outputs


@ingredient.capture
def build(grayscale, rows, cols, blocks, layers, optimizer, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          _log=None, *args, **kwargs):
    if 'name' in kwargs:
        _log.info('Build CNN model [%s]' % kwargs['name'])
    else:
        _log.info('Build CNN model')

    filters = 1 if grayscale else 3
    if K.image_data_format() == 'channels_first':
        input_shape = (filters, rows, cols)
    else:
        input_shape = (rows, cols, filters)

    inputs = Input(shape=input_shape, name='input')
    if 'noise_config' in layers and layers['noise_config']:
        x = deserialize_layer(layers['noise_config'])(inputs)
    else:
        x = inputs

    if 'filters' in layers and type(layers['filters']) == list:
        assert len(layers['filters']) == blocks

    shortcuts = []
    for i in range(blocks):
        filters = None
        if 'filters' in layers:
            if type(layers['filters']) == list:
                filters = layers['filters'][i]
            else:
                filters = layers['filters']

        pool = i != blocks - 1
        if 'bn_config' in layers and layers['bn_config']:
            x = block2d_bn(x, pool=pool, shortcuts=shortcuts, filters=filters)
        else:
            x = block2d(x, pool=pool, shortcuts=shortcuts, filters=filters)

        if i != blocks - 1:
            rows = math.ceil(rows / layers['strides'][0])
            cols = math.ceil(cols / layers['strides'][1])

    # outputs
    outs, loss, metrics = outputs(x, rows=int(rows), cols=int(cols),
                                  shortcuts=shortcuts)

    # Model
    model = Model(inputs=inputs, outputs=outs,
                  name=kwargs['name'] if 'name' in kwargs else 'cnn')
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics, loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model


@ingredient.capture(prefix='layers')
def conv2d(x, conv2d_config, dropout=None, filters=None):
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)

    if filters:
        conv2d_config = dict(conv2d_config, **{'filters': filters})
    return Conv2D.from_config(conv2d_config)(x)


@ingredient.capture(prefix='layers')
def conv2d_bn(x, bn_config, conv2d_config, activation, dropout=None,
              filters=None):
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)

    if filters:
        conv2d_config = dict(conv2d_config, **{'filters': filters})
    x = Conv2D.from_config(conv2d_config)(x)
    x = BatchNormalization.from_config(bn_config)(x)
    return Activation(activation)(x)


@ingredient.capture(prefix='layers')
def block2d(inputs, N, conv2d_config, strides, pool, dropout=None,
            filters=None, *args, **kwargs):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters=filters)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        if dropout and dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)

        conf = dict(conv2d_config, **{'strides': strides})
        if filters:
            conf['filters'] = filters
        x = Conv2D.from_config(conf)(x)
        if dropout and dropout['t'] == 'blockwise':
            x = deserialize_layer(dropout)(x)
    return x


@ingredient.capture(prefix='layers')
def block2d_bn(inputs, N, bn_config, conv2d_config, activation, strides, pool,
               dropout=None, filters=None, *args, **kwargs):
    for j in range(N):
        x = conv2d_bn(inputs if j == 0 else x, filters=filters)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        if dropout and dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)

        conf = dict(conv2d_config, **{'strides': strides})
        if filters:
            conf['filters'] = filters
        x = Conv2D.from_config(conf)(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
        if dropout and dropout['t'] == 'blockwise':
            x = deserialize_layer(dropout)(x)
    return x


@ingredient.capture(prefix='layers')
def upblock2d(inputs, N, conv2d_config, strides, transpose, dropout=None,
              filters=None, *args, **kwargs):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters=filters)

    if transpose:
        if dropout and dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)

        conf = dict(conv2d_config, **{'strides': strides})
        if filters:
            conf['filters'] = filters
        x = Conv2DTranspose.from_config(conf)(x)
        if dropout and dropout['t'] == 'blockwise':
            x = deserialize_layer(dropout)(x)
    return x


@ingredient.capture(prefix='layers')
def upblock2d_bn(inputs, filters, N, bn_config, conv2d_config, activation,
                 strides, transpose, dropout=None, *args, **kwargs):
    for j in range(N):
        x = conv2d_bn(inputs if j == 0 else x, filters=filters)

    if transpose:
        if dropout and dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)

        conf = dict(conv2d_config, **{'strides': strides})
        if filters:
            conf['filters'] = filters
        x = Conv2DTranspose.from_config(conf)(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
        if dropout and dropout['t'] == 'blockwise':
            x = deserialize_layer(dropout)(x)
    return x
