# -*- coding: utf-8 -*-

import math

from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.layers import cnn, densely
from ingredients.models import ingredients


@ingredients.config
def config():
    encoder = {
        'layers': {
            'nb_dense': 1,
            'dense_config': {
                'units': 300,
                'kernel_initializer': 'lecun_uniform',
                'activation': 'selu'
            }
        },
        'net_type': 'fc',
        'loss': 'mse',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy']
    }
    decoder = {
        'layers': {
            'nb_dense': 1,
            'dense_config': {
                'kernel_initializer': 'lecun_uniform',
                'activation': 'selu'
            }
        },
        'net_type': 'fc',
        'loss': 'mse',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy']
    }
    autoencoder = {
        'loss': 'mse',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy']
    }


@ingredients.capture(prefix='autoencoder')
def build(input_shape, loss, optimizer, metrics, *args, **kwargs):
    print('Building AutoEncoder')

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.layers[-1].output_shape[1:], input_shape)

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
    assert net_type in ['fc']

    print('Building Encoder [net type: %s]' % net_type)

    if net_type == 'fc':
        inputs = Input(input_shape, name='input')
        for i in range(layers['nb_dense']):
            x = Dense.from_config(layers['dense_config'])(inputs if i == 0 else x)

    # Model
    model = Model(inputs=inputs, outputs=x, name='encoder')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model


@ingredients.capture(prefix='decoder')
def build_decoder(input_shape, output_shape, net_type, layers, loss, optimizer,
                  metrics):
    assert net_type in ['fc']

    print('Building Decoder [net type: %s]' % net_type)

    if net_type == 'fc':
        inputs = Input(input_shape, name='input')
        for i in range(layers['nb_dense']):
            config = layers['dense_config'].copy()
            if i == layers['nb_dense'] - 1:
                config['units'] = output_shape[0]
            x = Dense.from_config(config)(inputs if i == 0 else x)

    # Model
    model = Model(inputs=inputs, outputs=x, name='decoder')
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return model
