# -*- coding: utf-8 -*-

import math

from copy import deepcopy
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
        _log.info('Build DenselyCNN model [%s]' % kwargs['name'])
    else:
        _log.info('Build DenselyCNN model')

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

    shortcuts = []
    for i in range(blocks):
        pool = i != blocks - 1
        if 'bn_config' in layers and layers['bn_config']:
            x, filters = block2d_bn(x, filters, pool=pool, shortcuts=shortcuts)
        else:
            x, filters = block2d(x, filters, pool=pool, shortcuts=shortcuts)

        if i != blocks - 1:
            rows = math.ceil(rows / layers['strides'][0])
            cols = math.ceil(cols / layers['strides'][1])

    # outputs
    outs, loss, metrics = outputs(x, rows=int(rows), cols=int(cols),
                                  shortcuts=shortcuts)

    # Model
    model = Model(inputs=inputs, outputs=outs,
                  name=kwargs['name'] if 'name' in kwargs else 'densely')
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics, loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model


@ingredient.capture(prefix='layers')
def conv2d(x, k, bottleneck, bottleneck2d_config, conv2d_config, dropout=None):
    if bottleneck:
        x = Conv2D.from_config(dict(bottleneck2d_config,
                                    **{'filters': bottleneck * k}))(x)
    x = Conv2D.from_config(dict(conv2d_config, **{'filters': k}))(x)
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)
    return x


@ingredient.capture(prefix='layers')
def conv2d_bn(x, k, bottleneck, bn_config, bottleneck2d_config, conv2d_config,
              activation, dropout=None):
    if bottleneck:
        x = Conv2D.from_config(dict(bottleneck2d_config,
                                    **{'filters': bottleneck * k}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
    x = Conv2D.from_config(dict(conv2d_config, **{'filters': k}))(x)
    x = BatchNormalization.from_config(bn_config)(x)
    x = Activation(activation)(x)
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)
    return x


@ingredient.capture(prefix='layers')
def block2d(inputs, filters, N, k, bottleneck, bottleneck2d_config,
            conv2d_config, strides, theta, pool, concat_axis, dropout=None,
            *args, **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d(inputs if j == 0 else x))
        x = concatenate([inputs] + convs, axis=concat_axis)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config(dict(bottleneck2d_config,
                                        **{'filters': filters}))(x)
        x = Conv2D.from_config(dict(conv2d_config,
                                    **{'filters': filters,
                                       'strides': strides}))(x)
        if dropout and dropout['t'] == 'blockwise' or \
                dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)
    return x, filters


@ingredient.capture(prefix='layers')
def block2d_bn(inputs, filters, N, k, bottleneck, bn_config,
               bottleneck2d_config, conv2d_config, activation, strides, theta,
               pool, concat_axis, dropout=None, *args, **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d_bn(inputs if j == 0 else x))
        x = concatenate([inputs] + convs, axis=concat_axis)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config(dict(bottleneck2d_config,
                                        **{'filters': filters}))(x)
            x = BatchNormalization.from_config(bn_config)(x)
            x = Activation(activation)(x)
        x = Conv2D.from_config(dict(conv2d_config,
                                    **{'filters': filters,
                                       'strides': strides}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
        if dropout and dropout['t'] == 'blockwise' or \
                dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)
    return x, filters


@ingredient.capture(prefix='layers')
def upblock2d(inputs, filters, N, k, bottleneck, bottleneck2d_config,
              conv2d_config, strides, theta, transpose, concat_axis,
              dropout=None, *args, **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d(inputs if j == 0 else x))
        x = concatenate([inputs] + convs, axis=concat_axis)

    if transpose:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config(dict(bottleneck2d_config,
                                        **{'filters': filters}))(x)
        x = Conv2DTranspose.from_config(dict(conv2d_config,
                                             **{'filters': filters,
                                                'strides': strides}))(x)
        if dropout and dropout['t'] == 'blockwise' or \
                dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)
    return x, filters


@ingredient.capture(prefix='layers')
def upblock2d_bn(inputs, filters, N, k, bottleneck, bn_config,
                 bottleneck2d_config, conv2d_config, activation, strides,
                 theta, transpose, concat_axis, dropout=None, *args, **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d_bn(inputs if j == 0 else x))
        x = concatenate([inputs] + convs, axis=concat_axis)

    if transpose:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config(dict(bottleneck2d_config,
                                        **{'filters': filters}))(x)
            x = BatchNormalization.from_config(bn_config)(x)
            x = Activation(activation)(x)
        x = Conv2DTranspose.from_config(dict(conv2d_config,
                                             **{'filters': filters,
                                                'strides': strides}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
        if dropout and dropout['t'] == 'blockwise' or \
                dropout['t'] == 'layerwise':
            x = deserialize_layer(dropout)(x)
    return x, filters
