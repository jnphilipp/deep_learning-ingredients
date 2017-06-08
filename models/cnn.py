# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Activation, Conv2D, Flatten, GaussianNoise, Input
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.layers import cnn, densely
from ingredients.models import ingredients


@ingredients.config
def config():
    cnn = {
        'layers': {
            'N': 10,
            'k': 12,
            'bottleneck': 4,
            'gaussian_noise_config': {
                'stddev': 0.3
            },
            'bn_config': {
                'axis': 1
            },
            'bottleneck2d_config': {
                'kernel_size': (1, 1),
                'padding': 'same'
            },
            'conv2d_config': {
                'kernel_size': (3, 3),
                'padding': 'same'
            },
            'strides': (2, 2),
            'activation': 'tanh',
            'theta': 0.5,
            'concat_axis': 1
        },
        'net_type': 'densely',
        'loss': 'categorical_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy']
    }


@ingredients.capture(prefix='cnn')
def build(nb_classes, net_type, *args, **kwargs):
    assert net_type in ['cnn', 'densely']

    print('Building CNN [net type: %s]...' % net_type)
    if net_type == 'cnn':
        return build_cnn(nb_classes=nb_classes, **kwargs)
    elif net_type == 'densely':
        return build_densely(nb_classes=nb_classes, **kwargs)


@ingredients.capture(prefix='cnn')
def build_cnn(grayscale, rows, cols, blocks, nb_classes, layers, loss,
              optimizer, metrics):
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

    for i in range(blocks):
        if 'bn_config' in layers and layers['bn_config']:
            x = cnn.block2d_bn(x, pool=i != blocks - 1, **layers)
        else:
            x = cnn.block2d(x, pool=i != blocks - 1, **layers)
        if i != blocks - 1:
            rows /= layers['strides'][0]
            cols /= layers['strides'][1]

    # softmax
    x = Conv2D(nb_classes, (int(rows), int(cols)))(x)
    x = Flatten()(x)
    predictions = Activation('softmax', name='p')(x)

    # Model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model


@ingredients.capture(prefix='cnn')
def build_densely(grayscale, rows, cols, blocks, nb_classes, layers, loss,
                  optimizer, metrics):
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

    for i in range(blocks):
        if 'bn_config' in layers and layers['bn_config']:
            x, filters = densely.block2d_bn(x, filters, pool=i != blocks - 1,
                                            **layers)
        else:
            x, filters = densely.block2d(x, filters, pool=i != blocks - 1,
                                         **layers)
        if i != blocks - 1:
            rows /= layers['strides'][0]
            cols /= layers['strides'][1]

    # softmax
    x = Conv2D(nb_classes, (int(rows), int(cols)))(x)
    x = Flatten()(x)
    predictions = Activation('softmax', name='p')(x)

    # Model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model
