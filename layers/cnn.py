# -*- coding: utf-8 -*-

from keras.layers import Conv2D, Conv2DTranspose
from ingredients.layers import ingredients


@ingredients.config
def config():
    bn_config = {'axis': 1}
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
def block2d(inputs, filters, N, conv2d_config, activation, strides, pool):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters=filters)

    if pool:
        return Conv2D.from_config(dict(conv2d_config,
                                       **{'filters': filters,
                                          'strides': strides,
                                          'activation': activation}))(x)
    else:
        return x


@ingredients.capture
def upblock2d(inputs, filters, N, conv2d_config, activation, strides, transpose):
    for j in range(N):
        x = conv2d(inputs if j == 0 else x, filters=filters)

    if transpose:
        return Conv2DTranspose.from_config(dict(conv2d_config,
                                                **{'filters': filters,
                                                   'strides': strides,
                                                   'activation': activation}))(x)
    else:
        return x
