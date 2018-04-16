# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import (concatenate, Activation, AlphaDropout, Conv2D,
                          Conv2DTranspose as Conv2DT, Flatten, GaussianNoise,
                          Input)
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
                'axis': 1 if K.image_data_format() == 'channels_first' else -1
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
            'p_activation': 'softmax',
            'theta': 0.5,
            'concat_axis': 1 if K.image_data_format() == 'channels_first'
            else -1
        },
        'net_type': 'densely',
        'outputs': ['class'],
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
def build_cnn(grayscale, rows, cols, blocks, nb_classes, layers, outputs, loss,
              optimizer, metrics, *args, **kwargs):
    assert set(outputs).issubset(['class', 'image', 'mask'])

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
        if 'bn_config' in layers and layers['bn_config']:
            x = cnn.block2d_bn(x, pool=i != blocks - 1, shortcuts=shortcuts,
                               **layers)
        else:
            x = cnn.block2d(x, pool=i != blocks - 1, shortcuts=shortcuts,
                            **layers)
        if 'alpha_dropout_config' in layers and layers['alpha_dropout_config']:
            x = AlphaDropout.from_config(layers['alpha_dropout_config'])(x)
        if i != blocks - 1:
            rows = math.ceil(rows / layers['strides'][0])
            cols = math.ceil(cols / layers['strides'][1])

    # output
    otensors = []
    for i, output in enumerate(outputs):
        if output == 'class':
            x = Conv2D(nb_classes, (int(rows), int(cols)))(x)
            x = Flatten()(x)
            otensors.append(Activation(layers['p_activation'], name='p')(x))
        elif output == 'image':
            otensors.append(Conv2D(1 if grayscale else 3, (1, 1),
                            padding='same', activation='sigmoid')(x))
        elif output == 'mask':
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
                                   'strides': layers['strides'],
                                   'activation': layers['activation']})
                    x = Conv2DT.from_config(conf)(x)
            otensors.append(Conv2D.from_config(dict(layers['conv2d_config'],
                                                    **{'filters': 1,
                                                       'activation': 'sigmoid',
                                                       'name': 'mask'}))(x))

    # Model
    model = Model(inputs=inputs, outputs=otensors,
                  name=kwargs['name'] if 'name' in kwargs else 'cnn')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model


@ingredients.capture(prefix='cnn')
def build_densely(grayscale, rows, cols, blocks, nb_classes, layers, loss,
                  outputs, optimizer, metrics, *args, **kwargs):
    assert set(outputs).issubset(['class', 'image', 'mask'])

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
        if 'bn_config' in layers and layers['bn_config']:
            x, filters = densely.block2d_bn(x, filters, pool=i != blocks - 1,
                                            shortcuts=shortcuts, **layers)
        else:
            x, filters = densely.block2d(x, filters, pool=i != blocks - 1,
                                         shortcuts=shortcuts, **layers)
        if i != blocks - 1:
            rows = math.ceil(rows / layers['strides'][0])
            cols = math.ceil(cols / layers['strides'][1])

    # output
    otensors = []
    for i, output in enumerate(outputs):
        if output == 'class':
            x = Conv2D(nb_classes, (int(rows), int(cols)))(x)
            x = Flatten()(x)
            otensors.append(Activation(layers['p_activation'], name='p')(x))
        elif output == 'image':
            otensors.append(Conv2D(1 if grayscale else 3, (1, 1),
                            padding='same', activation='sigmoid')(x))
        elif output == 'mask':
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
                                   'strides': layers['strides'],
                                   'activation': layers['activation']})
                    x = Conv2DT.from_config(conf)(x)
            otensors.append(Conv2D.from_config(dict(layers['conv2d_config'],
                                                    **{'filters': 1,
                                                       'activation': 'sigmoid',
                                                       'name': 'mask'}))(x))

    # Model
    model = Model(inputs=inputs, outputs=otensors,
                  name=kwargs['name'] if 'name' in kwargs else 'cnn')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model
