# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.layers import cnn, densely
from ingredients.models import ingredients


@ingredients.config
def config():
    encoder = {
        'layers': {
            'nb_layers': 1,
            'dense_config': {
                'units': 300,
                'kernel_initializer': 'lecun_uniform',
                'activation': 'selu'
            }
        },
        'net_type': 'fc',
        'loss': 'binary_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['binary_accuracy']
    }
    decoder = {
        'layers': {
            'nb_layers': 1,
            'dense_config': {
                'kernel_initializer': 'lecun_uniform',
                'activation': 'selu'
            }
        },
        'net_type': 'fc',
        'loss': 'binary_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['binary_accuracy']
    }
    autoencoder = {
        'loss': 'binary_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['binary_accuracy']
    }


@ingredients.capture(prefix='autoencoder')
def build(input_shape, loss, optimizer, metrics, *args, **kwargs):
    print('Building AutoEncoder')

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder, input_shape)

    inputs = Input(shape=input_shape, name='input')
    x = encoder(inputs)
    outputs = decoder(x)

    # Model
    model = Model(inputs=inputs, outputs=outputs,
                  name=kwargs['name'] if 'name' in kwargs else 'autoencoder')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model


@ingredients.capture(prefix='encoder')
def build_encoder(input_shape, net_type, layers, loss, optimizer, metrics):
    assert net_type in ['conv2d', 'fc']
    print('Building Encoder [net type: %s]' % net_type)

    inputs = Input(input_shape, name='input')
    x = inputs
    if net_type == 'conv2d':
        for i in range(layers['nb_layers']):
            config = layers['conv2d_config'].copy()
            if i == layers['nb_layers'] - 1:
                if K.image_data_format() == 'channels_first':
                    config['strides'] = (int(int(x.shape[2]) / 1),
                                         int(int(x.shape[3]) / 1))
                elif K.image_data_format() == 'channels_last':
                    config['strides'] = (int(int(x.shape[1]) / 1),
                                         int(int(x.shape[2]) / 1))
            x = Conv2D.from_config(config)(x)
        x = Flatten()(x)
    elif net_type == 'fc':
        for i in range(layers['nb_layers']):
            x = Dense.from_config(layers['dense_config'])(x)

    # Model
    model = Model(inputs=inputs, outputs=x, name='encoder')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model


@ingredients.capture(prefix='decoder')
def build_decoder(encoder, output_shape, net_type, layers, loss, optimizer,
                  metrics):
    assert net_type in ['conv2dtranspose', 'fc']
    print('Building Decoder [net type: %s]' % net_type)

    input_shape = encoder.layers[-1].output_shape[1:]
    inputs = Input(input_shape, name='input')
    x = inputs
    if net_type == 'conv2dtranspose':
        if K.image_data_format() == 'channels_first':
            x = Reshape((input_shape[-1], 1, 1))(x)
        elif K.image_data_format() == 'channels_last':
            x = Reshape((1, 1, input_shape[-1]))(x)
        for i in range(layers['nb_layers']):
            config = layers['conv2dtranspose_config'].copy()
            if i == 0:
                config['strides'] = encoder.layers[-2].strides
            elif i == layers['nb_layers'] - 1:
                if K.image_data_format() == 'channels_first':
                    config['filters'] = output_shape[0]
                elif K.image_data_format() == 'channels_last':
                    config['filters'] = output_shape[-1]
            x = Conv2DTranspose.from_config(config)(x)
    elif net_type == 'fc':
        for i in range(layers['nb_layers']):
            config = layers['dense_config'].copy()
            if i == layers['nb_layers'] - 1:
                config['units'] = output_shape[0]
            x = Dense.from_config(config)(x)

    # Model
    model = Model(inputs=inputs, outputs=x, name='decoder')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model
