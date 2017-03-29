# -*- coding: utf-8 -*-

from keras.layers import concatenate, Activation, BatchNormalization, Conv2D, Conv2DTranspose
from ingredients.layers import ingredients


@ingredients.config
def config():
    N = 10
    k = 12
    bottleneck = 4
    bn_config = {'axis': 1}
    bottleneck_config = {'kernel_size': (1, 1),
                    'padding': 'same'
    }
    conv_config = {'kernel_size': (3, 3),
                    'padding': 'same'
    }
    strides = (2, 2)
    activation = 'tanh'
    theta = 0.5
    concat_axis = 1


@ingredients.capture
def conv2d(x, k, bottleneck, bottleneck_config, conv_config, activation):
    if bottleneck:
        x = Conv2D.from_config({**bottleneck_config, **{'filters': bottleneck * k, 'activation': activation}})(x)
    return Conv2D.from_config({**conv_config, **{'filters': k, 'activation': activation}})(x)


@ingredients.capture
def conv2d_bn(x, k, bottleneck, bn_config, bottleneck_config, conv_config, activation):
    if bottleneck:
        x = Conv2D.from_config({**bottleneck_config, **{'filters': bottleneck * k}})(x)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
    x = Conv2D.from_config({**conv_config, **{'filters': k}})(x)
    x = BatchNormalization.from_config(bn_config)(x)
    return Activation(activation)(x)


@ingredients.capture
def block2d_bn(inputs, filters, N, k, bottleneck, bn_config, bottleneck_config, conv_config, activation, strides, theta, pool, concat_axis):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d_bn(inputs if j == 0 else x))
        x = concatenate([inputs] + convs, axis=concat_axis)

    if pool:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config({**bottleneck_config, **{'filters': filters}})(x)
            x = BatchNormalization.from_config(bn_config)(x)
            x = Activation(activation)(x)
        x = Conv2D.from_config({**conv_config, **{'filters': filters, 'strides': strides}})(x)
        x = BatchNormalization.from_config(bn_config)(x)
        return Activation(activation)(x), filters
    else:
        return x, filters


@ingredients.capture
def upblock2d(inputs, filters, N, k, bottleneck, bottleneck_config, conv_config, activation, strides, theta, transpose):
    convs = []
    for j in range(N):
        filters += k
        convs.append(conv2d(inputs if j == 0 else x))
        x = concatenate([inputs] + convs, axis=1)

    if transpose:
        filters = int(filters * theta)
        if bottleneck:
            x = Conv2D.from_config({**bottleneck_config, **{'filters': filters, 'activation': activation}})(x)
        return Conv2DTranspose.from_config({**conv_config, **{'filters': filters, 'strides': strides, 'activation': activation}})(x), filters
    else:
        return x, filters
