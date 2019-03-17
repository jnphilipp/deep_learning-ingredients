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
def build(grayscale, rows, cols, blocks, layers, optimizer,
          connection_type='base', loss_weights=None, sample_weight_mode=None,
          weighted_metrics=None, target_tensors=None, _log=None, *args,
          **kwargs):
    assert connection_type in ['base', 'densely']
    if connection_type == 'base':
        connection_name = ''
    elif connection_type == 'densely':
        connection_name = 'Densely'

    if 'name' in kwargs:
        name = kwargs['name']
        _log.info(f'Build {connection_name}CNN model [{name}]')
    else:
        if connection_type == 'base':
            name = 'cnn'
        elif connection_type == 'densely':
            name = 'densely-cnn'
        _log.info(f'Build {connection_name}CNN model')

    nb_filters = 1 if grayscale else 3
    if K.image_data_format() == 'channels_first':
        input_shape = (nb_filters, rows, cols)
    else:
        input_shape = (rows, cols, nb_filters)

    inputs = Input(shape=input_shape, name='input')
    if 'noise' in layers and layers['noise']:
        x = deserialize_layer(layers['noise'])(inputs)
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

        x, cols, rows = block(x, do_pooling=i != blocks - 1, filters=filters,
                              shortcuts=shortcuts, cols=cols, rows=rows,
                              nb_filters=nb_filters,
                              connection_type=connection_type)

    # outputs
    if 'outputs' in kwargs:
        outs, loss, metrics = outputs(x, rows=int(rows), cols=int(cols),
                                      shortcuts=shortcuts,
                                      outputs=kwargs['outputs'])
    else:
        outs, loss, metrics = outputs(x, rows=int(rows), cols=int(cols),
                                      shortcuts=shortcuts)

    # Model
    model = Model(inputs=inputs, outputs=outs, name=name)
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics, loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model


@ingredient.capture(prefix='layers')
def layer(x, batchnorm=None, bottleneck2d=None, conv2d={}, dropout=None,
          filters=None, k=None):
    if batchnorm is not None:
        if bottleneck2d is not None:
            bottleneck_activation = bottleneck2d['activation']
            bottleneck2d = dict(bottleneck2d, **{'activation': 'linear'})

        conv_activation = conv2d['activation']
        conv2d = dict(conv2d, **{'activation': 'linear'})

    if bottleneck2d is not None:
        assert k is not None

        factor = bottleneck2d['filters']
        x = Conv2D.from_config(dict(bottleneck2d,
                                    **{'filters': factor * k}))(x)
        if batchnorm is not None:
            x = BatchNormalization.from_config(batchnorm)(x)
            x = Activation(bottleneck_activation)(x)

    if filters is not None:
        conv2d = dict(conv2d, **{'filters': filters})
    elif k is not None:
        conv2d = dict(conv2d, **{'filters': k})

    x = Conv2D.from_config(conv2d)(x)
    if batchnorm is not None:
        x = BatchNormalization.from_config(batchnorm)(x)
        x = Activation(conv_activation)(x)
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)
    return x


@ingredient.capture(prefix='layers')
def block(inputs, N, cols, rows, connection_type='base', do_pooling=True,
          batchnorm=None, bottleneck2d=None, conv2d={}, pooling={},
          concat_axis=1 if K.image_data_format() == 'channels_first' else -1,
          dropout=None, theta=None, filters=None, k=None, *args, **kwargs):
    assert connection_type in ['base', 'densely']
    if connection_type == 'densely':
        assert concat_axis is not None
    nb_filters = kwargs['nb_filters'] if 'nb_filters' in kwargs else 0

    convs = []
    for j in range(N):
        if k is not None:
            nb_filters += k

        if connection_type == 'base':
            x = layer(inputs if j == 0 else x, filters=filters)
        elif connection_type == 'densely':
            convs.append(layer(inputs if j == 0 else x))
            x = concatenate([inputs] + convs, axis=concat_axis)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, nb_filters))

    if do_pooling:
        if theta is not None:
            nb_filters = int(nb_filters * theta)

        if batchnorm is not None:
            if bottleneck2d is not None:
                bottleneck_activation = bottleneck2d['activation']
                bottleneck2d = dict(bottleneck2d, **{'activation': 'linear'})

            if pooling['class_name'].lower() == 'conv2d':
                pooling_activation = conv2d['activation']
                pooling['config'] = dict(pooling['config'],
                                         **{'activation': 'linear'})

        if bottleneck2d is not None:
            assert k is not None

            if theta is not None:
                bottleneck2d = dict(bottleneck2d, **{'filters': nb_filters})
            else:
                factor = bottleneck2d['filters']
                bottleneck2d = dict(bottleneck2d, **{'filters': factor * k})
            x = Conv2D.from_config(bottleneck2d)(x)
            if batchnorm is not None:
                x = BatchNormalization.from_config(batchnorm)(x)
                x = Activation(bottleneck_activation)(x)

        if filters is not None:
            pooling['config'] = dict(pooling['config'], **{'filters': filters})
        elif k is not None:
            if theta is not None:
                pooling['config'] = dict(pooling['config'],
                                         **{'filters': nb_filters})
            else:
                pooling['config'] = dict(pooling['config'], **{'filters': k})

        x = deserialize_layer(pooling)(x)
        if batchnorm is not None and pooling['class_name'].lower() == 'conv2d':
            x = BatchNormalization.from_config(batchnorm)(x)
            x = Activation(pooling_activation)(x)
        if dropout and dropout['t'] in ['blockwise', 'layerwise']:
            x = deserialize_layer(dropout)(x)

        rows = math.ceil(rows / pooling['config']['strides'][0])
        cols = math.ceil(cols / pooling['config']['strides'][1])
    return x, cols, rows


@ingredient.capture(prefix='layers')
def upblock(inputs, N, cols, rows, batchnorm=None, bottleneck2d=None,
            conv2d={}, conv2dt={}, dropout=None, do_transpose=True,
            concat_axis=1 if K.image_data_format() == 'channels_first' else -1,
            theta=None, filters=None, k=None, *args, **kwargs):
    assert connection_type in ['base', 'densely']
    if connection_type == 'densely':
        assert concat_axis
    nb_filters = kwargs['nb_filters'] if 'nb_filters' in kwargs else 0

    convs = []
    for j in range(N):
        if k is not None:
            nb_filters += k

        if connection_type == 'base':
            x = layer(inputs if j == 0 else x, filters=filters)
        elif connection_type == 'densely':
            convs.append(layer(inputs if j == 0 else x))
            x = concatenate([inputs] + convs, axis=concat_axis)

    if do_transpose:
        if theta is not None:
            nb_filters = int(nb_filters * theta)

        if batchnorm is not None:
            if bottleneck2d is not None:
                bottleneck_activation = bottleneck2d['activation']
                bottleneck2d = dict(bottleneck2d, **{'activation': 'linear'})

            convt_activation = conv2dt['activation']
            conv2dt = dict(conv2dt, **{'activation': 'linear'})

        if bottleneck2d is not None:
            assert k is not None

            if theta is not None:
                bottleneck2d = dict(bottleneck2d, **{'filters': nb_filters})
            else:
                factor = bottleneck2d['filters']
                bottleneck2d = dict(bottleneck2d, **{'filters': factor * k})
            x = Conv2D.from_config(bottleneck2d)(x)
            if batchnorm is not None:
                x = BatchNormalization.from_config(batchnorm)(x)
                x = Activation(bottleneck_activation)(x)

        if filters is not None:
            conv2dt = dict(conv2dt, **{'filters': filters})
        elif k is not None:
            if theta is not None:
                conv2dt = dict(conv2dt, **{'filters': nb_filters})
            else:
                conv2dt = dict(conv2dt, **{'filters': k})

        x = Conv2DTranspose.from_config(conv2dt)(x)
        if batchnorm is not None:
            x = BatchNormalization.from_config(batchnorm)(x)
            x = Activation(conv2dt)(x)
        if dropout and dropout['t'] in ['blockwise', 'layerwise']:
            x = deserialize_layer(dropout)(x)
    return x
