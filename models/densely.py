# -*- coding: utf-8 -*-

import math

from copy import deepcopy
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.models import ingredients


@ingredients.capture
def build(grayscale, rows, cols, blocks, layers, outputs, optimizer, _log,
          loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
          target_tensors=None, *args, **kwargs):
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
    if 'gaussian_noise_config' in layers and layers['gaussian_noise_config']:
        x = GaussianNoise.from_config(layers['gaussian_noise_config'])(inputs)
    else:
        x = inputs

    shortcuts = []
    for i in range(blocks):
        pool = i != blocks - 1
        if 'bn_config' in layers and layers['bn_config']:
            x, filters = block2d_bn(x, filters, pool=pool, shortcuts=shortcuts)
        else:
            x, filters = block2d(x, filters, pool=pool, shortcuts=shortcuts)

        if 'alpha_dropout_config' in layers and layers['alpha_dropout_config']:
            x = AlphaDropout.from_config(layers['alpha_dropout_config'])(x)

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
            x = Conv2D.from_config(dict(layers['conv2d_config'],
                                   **{'filters': output['nb_classes'],
                                      'kernel_size': (int(rows), int(cols)),
                                      'padding': 'valid'}))(x)
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
                  name=kwargs['name'] if 'name' in kwargs else 'densely')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model


@ingredients.capture(prefix='layers')
def conv2d(x, k, bottleneck, bottleneck2d_config, conv2d_config):
    if bottleneck:
        x = Conv2D.from_config(dict(bottleneck2d_config,
                                    **{'filters': bottleneck * k}))(x)
    return Conv2D.from_config(dict(conv2d_config, **{'filters': k}))(x)


@ingredients.capture(prefix='layers')
def conv2d_bn(x, k, bottleneck, bn_config, bottleneck2d_config, conv2d_config,
              activation):
    if bottleneck:
        x = Conv2D.from_config(dict(bottleneck2d_config,
                                    **{'filters': bottleneck * k}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
    x = Conv2D.from_config(dict(conv2d_config, **{'filters': k}))(x)
    x = BatchNormalization.from_config(bn_config)(x)
    return Activation(activation)(x)


@ingredients.capture(prefix='layers')
def block2d(inputs, filters, N, k, bottleneck, bottleneck2d_config,
            conv2d_config, strides, theta, pool, concat_axis, *args, **kwargs):
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
        return Conv2D.from_config(dict(conv2d_config,
                                       **{'filters': filters,
                                          'strides': strides}))(x), filters
    else:
        return x, filters


@ingredients.capture(prefix='layers')
def block2d_bn(inputs, filters, N, k, bottleneck, bn_config,
               bottleneck2d_config, conv2d_config, activation, strides, theta,
               pool, concat_axis, *args, **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d_bn(inputs if j == 0 else x, k, bottleneck,
                               bn_config, bottleneck2d_config, conv2d_config,
                               activation))
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
        return Activation(activation)(x), filters
    else:
        return x, filters


@ingredients.capture(prefix='layers')
def upblock2d(inputs, filters, N, k, bottleneck, bottleneck2d_config,
              conv2d_config, strides, theta, transpose, concat_axis, *args,
              **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d(inputs if j == 0 else x, k, bottleneck,
                            bottleneck2d_config, conv2d_config))
        x = concatenate([inputs] + convs, axis=concat_axis)

    if transpose:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config(dict(bottleneck2d_config,
                                        **{'filters': filters}))(x)
        conf = dict(conv2d_config, **{'filters': filters, 'strides': strides})
        return Conv2DTranspose.from_config(conf)(x), filters
    else:
        return x, filters


@ingredients.capture(prefix='layers')
def upblock2d_bn(inputs, filters, N, k, bottleneck, bn_config,
                 bottleneck2d_config, conv2d_config, activation, strides,
                 theta, transpose, concat_axis, *args, **kwargs):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d_bn(inputs if j == 0 else x, k, bottleneck,
                               bn_config, bottleneck2d_config, conv2d_config,
                               activation))
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
        return Activation(activation)(x), filters
    else:
        return x, filters
