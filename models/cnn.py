# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import *
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.models import ingredients


@ingredients.capture
def build(grayscale, rows, cols, blocks, layers, outputs, optimizer, _log,
          loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
          target_tensors=None, *args, **kwargs):
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
    if 'gaussian_noise_config' in layers and layers['gaussian_noise_config']:
        x = GaussianNoise.from_config(layers['gaussian_noise_config'])(inputs)
    else:
        x = inputs

    if type(layers['filters']) == list:
        assert len(layers['filters']) == blocks

    shortcuts = []
    for i in range(blocks):
        if type(layers['filters']) == list:
            conf = dict(layers, **{'filters': layers['filters'][i]})
        else:
            conf = layers

        pool = i != blocks - 1
        if 'bn_config' in layers and layers['bn_config']:
            x = block2d_bn(x, pool=pool, shortcuts=shortcuts, **conf)
        else:
            x = block2d(x, pool=pool, shortcuts=shortcuts, **conf)

        if i != blocks - 1:
            rows = math.ceil(rows / layers['strides'][0])
            cols = math.ceil(cols / layers['strides'][1])

    # outputs
    output_types = ['class', 'image', 'mask', 'vec']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = []
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics.append(output['metrics'])

        if output['t'] == 'class':
            conf = dict(layers['conv2d_config'],
                        **{'filters': output['nb_classes'],
                           'kernel_size': (int(rows), int(cols)),
                           'padding': 'valid'})
            x = Conv2D.from_config(conf)(x)
            x = Flatten()(x)
            outs.append(Activation(output['activation'], name='p')(x))
        elif output['t'] == 'image':
            conf = dict(layers['conv2d_config'],
                        **{'filters': 1 if output['grayscale'] else 3,
                           'kernel_size': (1, 1),
                           'padding': 'same',
                           'activation': output['activation']})
            outs.append(Conv2D.from_config(conf)(x))
        elif output['t'] == 'mask':
            for i in reversed(range(blocks)):
                shortcut = shortcuts[i][0]
                filters = shortcuts[i - 1 if i >= 0 else 0][1]
                if i is not blocks - 1:
                    x = concatenate([x, shortcut], axis=layers['concat_axis'])
                else:
                    x = shortcut
                if i > 0:
                    conf = dict(layers['conv2d_config'],
                                **{'filters': filters,
                                   'strides': layers['strides']})
                    x = Conv2DTranspose.from_config(conf)(x)
            outs.append(Conv2D.from_config(dict(layers['conv2d_config'],
                                                **{'filters': 1,
                                                   'activation': 'sigmoid',
                                                   'name': 'mask'}))(x))
        elif output['t'] == 'vec':
            outs.append(x)

    # Model
    model = Model(inputs=inputs, outputs=outs,
                  name=kwargs['name'] if 'name' in kwargs else 'cnn')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model


@ingredients.capture(prefix='layers')
def conv2d(x, filters, conv2d_config, dropout=None):
    x = Conv2D.from_config(dict(conv2d_config, **{'filters': filters}))(x)
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)
    return x


@ingredients.capture(prefix='layers')
def conv2d_bn(x, filters, bn_config, conv2d_config, activation, dropout=None):
    x = Conv2D.from_config(dict(conv2d_config, **{'filters': filters}))(x)
    x = BatchNormalization.from_config(bn_config)(x)
    x = Activation(activation)(x)
    if dropout and dropout['t'] == 'layerwise':
        x = deserialize_layer(dropout)(x)
    return x


@ingredients.capture(prefix='layers')
def block2d(inputs, filters, N, conv2d_config, strides, pool, dropout=None,
            *args, **kwargs):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters, conv2d_config)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        x = Conv2D.from_config(dict(conv2d_config,
                                    **{'filters': filters,
                                       'strides': strides}))(x)
        if dropout and (dropout['t'] == 'layerwise' or
                        dropout['t'] == 'blockwise'):
            x = deserialize_layer(dropout)(x)
        return x
    else:
        return x


@ingredients.capture(prefix='layers')
def block2d_bn(inputs, filters, N, bn_config, conv2d_config, activation,
               strides, pool, dropout=None, *args, **kwargs):
    for j in range(N):
        x = conv2d_bn(inputs if j == 0 else x, filters, bn_config,
                      conv2d_config, activation)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        x = Conv2D.from_config(dict(conv2d_config,
                                    **{'filters': filters,
                                       'strides': strides}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
        if dropout and (dropout['t'] == 'layerwise' or
                        dropout['t'] == 'blockwise'):
            x = deserialize_layer(dropout)(x)
        return x
    else:
        return x


@ingredients.capture(prefix='layers')
def upblock2d(inputs, filters, N, conv2d_config, strides, transpose,
              dropout=None, *args, **kwargs):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters, conv2d_config)

    if transpose:
        x = Conv2DTranspose.from_config(dict(conv2d_config,
                                             **{'filters': filters,
                                                'strides': strides}))(x)
        if dropout and (dropout['t'] == 'layerwise' or
                        dropout['t'] == 'blockwise'):
            x = deserialize_layer(dropout)(x)
        return x
    else:
        return x


@ingredients.capture(prefix='layers')
def upblock2d_bn(inputs, filters, N, bn_config, conv2d_config, activation,
                 strides, transpose, dropout=None, *args, **kwargs):
    for j in range(N):
        x = conv2d_bn(inputs if j == 0 else x, filters, bn_config,
                      conv2d_config, activation)

    if transpose:
        x = Conv2DTranspose.from_config(dict(conv2d_config,
                                        **{'filters': filters,
                                           'strides': strides}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
        if dropout and (dropout['t'] == 'layerwise' or
                        dropout['t'] == 'blockwise'):
            x = deserialize_layer(dropout)(x)
        return x
    else:
        return x
