# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose as Conv2DT)
from ingredients.layers import ingredients


@ingredients.config
def config():
    bn_config = {
        'axis': 1 if K.image_data_format() == 'channels_first' else -1
    }
    conv2d_config = {
        'kernel_size': (3, 3),
        'padding': 'same'
    }
    strides = (2, 2)
    activation = 'tanh'


@ingredients.capture
def conv2d(x, filters, conv2d_config, activation):
    return Conv2D.from_config(dict(conv2d_config,
                                   **{'filters': filters,
                                      'activation': activation}))(x)


@ingredients.capture
def conv2d_bn(x, filters, bn_config, conv2d_config, activation):
    x = Conv2D.from_config(dict(conv2d_config, **{'filters': filters}))(x)
    x = BatchNormalization.from_config(bn_config)(x)
    return Activation(activation)(x)


@ingredients.capture
def block2d(inputs, filters, N, conv2d_config, activation, strides, pool,
            *args, **kwargs):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters, conv2d_config, activation)

    if 'shortcuts' in kwargs:
        kwargs['shortcuts'].append((x, filters))

    if pool:
        return Conv2D.from_config(dict(conv2d_config,
                                       **{'filters': filters,
                                          'strides': strides,
                                          'activation': activation}))(x)
    else:
        return x


@ingredients.capture
def block2d_bn(inputs, filters, N, bn_config, conv2d_config, activation,
               strides, pool, *args, **kwargs):
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
        return Activation(activation)(x)
    else:
        return x


@ingredients.capture
def upblock2d(inputs, filters, N, conv2d_config, activation, strides,
              transpose, *args, **kwargs):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters, conv2d_config, activation)

    if transpose:
        return Conv2DT.from_config(dict(conv2d_config,
                                        **{'filters': filters,
                                           'strides': strides,
                                           'activation': activation}))(x)
    else:
        return x


@ingredients.capture
def upblock2d_bn(inputs, filters, N, bn_config, conv2d_config, activation,
                 strides, transpose, *args, **kwargs):
    for j in range(N):
        x = conv2d_bn(inputs if j == 0 else x, filters, bn_config,
                      conv2d_config, activation)

    if transpose:
        x = Conv2DT.from_config(dict(conv2d_config,
                                     **{'filters': filters,
                                        'strides': strides,
                                        'activation': activation}))(x)
        x = BatchNormalization.from_config(bn_config)(x)
        return Activation(activation)(x)
    else:
        return x
