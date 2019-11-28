# -*- coding: utf-8 -*-

import math

from logging import Logger
from tensorflow.keras.layers import concatenate, multiply
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Flatten, Input, Reshape)
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import List, Union

from . import ingredient
from .inputs import inputs
from .merge_layer import merge_layer
from .outputs import outputs


@ingredient.capture
def build(blocks: int, merge: dict, layers: dict, optimizer: Optimizer,
          _log: Logger, connection_type: str = 'base',
          loss_weights: Union[list, dict] = None,
          sample_weight_mode: str = None, weighted_metrics: list = None,
          target_tensors=None, *args, **kwargs) -> Model:
    assert connection_type in ['base', 'densely']
    if connection_type == 'base':
        connection_name = ''
    elif connection_type == 'densely':
        connection_name = 'Densely'

    if 'name' in kwargs:
        name = kwargs.pop('name')
        _log.info(f'Build {connection_name}CNN model [{name}]')
    else:
        if connection_type == 'base':
            name = 'cnn'
        elif connection_type == 'densely':
            name = 'densely-cnn'
        _log.info(f'Build {connection_name}CNN model')

    ins, xs = inputs()
    if 'depth' in merge and merge['depth'] == 0:
        xs = [merge_layer(xs)]

    if 'filters' in layers and type(layers['filters']) == list:
        assert len(layers['filters']) == blocks

    shortcuts: List[Tensor] = []
    tensors: dict = {}
    for i in range(blocks):
        for j, x in enumerate(xs):
            filters = None
            if 'filters' in layers:
                if type(layers['filters']) == list:
                    filters = layers['filters'][i]
                else:
                    filters = layers['filters']

            x = block(x, do_pooling=i != blocks - 1, filters=filters,
                      nb_filters=x._shape_val[-1], shortcuts=shortcuts,
                      connection_type=connection_type)

            xs[j] = x

    # outputs
    if 'outputs' in kwargs:
        outs, loss, metrics = outputs(xs, shortcuts=shortcuts,
                                      outputs=kwargs['outputs'])
    else:
        outs, loss, metrics = outputs(xs, shortcuts=shortcuts)

    # Model
    model = Model(inputs=ins, outputs=outs, name=name)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model


@ingredient.capture(prefix='layers')
def layer(x: Tensor, batchnorm: dict = None, bottleneck2d: dict = None,
          conv2d: dict = {}, dropout: dict = None, filters: int = None,
          k: int = None) -> Tensor:
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
def block(inputs: Tensor, N: int, connection_type: str = 'base',
          do_pooling: bool = True, attention2d: dict = None,
          batchnorm: dict = None, bottleneck2d: dict = None, conv2d: dict = {},
          pooling: dict = {}, concat_axis: int = -1, dropout: dict = None,
          theta: float = None, filters: int = None, k: int = None,
          *args, **kwargs) -> Tensor:
    assert connection_type in ['base', 'densely']
    if connection_type == 'densely':
        assert concat_axis is not None
    nb_filters = kwargs['nb_filters'] if 'nb_filters' in kwargs else 0
    pooling = pooling.copy()

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

    if attention2d is not None:
        w = Conv2D.from_config(dict(attention2d, **{'filters': 1}))(x)
        w_shape = w._shape_val[1:]
        w = Flatten()(w)
        w = Activation('softmax')(w)
        w = Reshape(w_shape)(w)
        x = multiply([x, w])

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

    return x


@ingredient.capture(prefix='layers')
def upblock(inputs: Tensor, N, cols, rows, attention2d=None, batchnorm=None,
            bottleneck2d=None, conv2d={}, conv2dt={}, dropout=None,
            do_transpose=True, concat_axis=-1,
            theta=None, filters=None, k=None, *args, **kwargs) -> Tensor:
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

    if attention2d is not None:
        w = Conv2D.from_config(dict(attention2d, **{'filters': nb_filters}))(x)
        w = Flatten()(w)
        w = Activation('softmax')(w)
        w = Reshape(x._keras_shape[1:])(w)
        x = multiply([x, w])

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
