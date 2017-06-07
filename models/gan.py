# -*- coding: utf-8 -*-

import operator

from functools import reduce
from keras import backend as K
from keras.layers import (multiply, Activation, Conv2D, Dense, Embedding,
                          Flatten, Input, Reshape, SpatialDropout1D)
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.layers import cnn, densely
from ingredients.models import ingredients


@ingredients.config
def config():
    generator = {
        'layers': {
            'N': 10,
            'k': 12,
            'bottleneck': 4,
            'bn_config': None,
            'bottleneck2d_config': {
                'kernel_size': (1, 1),
                'padding': 'same'
            },
            'conv2d_config': {
                'kernel_size': (3, 3),
                'padding': 'same'
            },
            'embedding_config': {
                'mask_zero': False,
                'name': 'embedding'
            },
            'strides': (2, 2),
            'activation': 'tanh',
            'theta': 0.5,
            'concat_axis': 1
        },
        'net_type': 'densely',
        'loss': 'binary_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {'lr': 0.0001, 'beta_1': 0.5, 'beta_2': 0.9}
        },
        'metrics': []
    }

    discriminator = {
        'layers': {
            'N': 10,
            'k': 12,
            'bottleneck': 4,
            'bn_config': {'axis': 1},
            'bottleneck2d_config': {
                'kernel_size': (1, 1),
                'padding': 'same'
            },
            'conv2d_config': {
                'kernel_size': (3, 3),
                'padding': 'same'
            },
            'embedding_config': {
                'output_dim': 128,
                'mask_zero': False,
                'name': 'embedding'
            },
            'strides': (2, 2),
            'dropout': 0.1,
            'activation': 'tanh',
            'f_activation': 'sigmoid',
            'theta': 0.5,
            'concat_axis': 1
        },
        'net_type': 'densely',
        'loss': ['binary_crossentropy', 'categorical_crossentropy'],
        'optimizer': {
            'class_name': 'adam',
            'config': {'lr': 0.0001, 'beta_1': 0.5, 'beta_2': 0.9}
        },
        'metrics': ['accuracy']
    }

    combined = {
        'loss': ['binary_crossentropy', 'categorical_crossentropy'],
        'optimizer': {
            'class_name': 'adam',
            'config': {'lr': 0.0001, 'beta_1': 0.5, 'beta_2': 0.9}
        },
        'metrics': ['accuracy']
    }


@ingredients.capture(prefix='generator')
def build_generator(nb_classes, latent_shape, blocks, net_type, layers, loss,
                    optimizer, metrics):
    assert net_type in ['cnn', 'densely']

    print('Building generator [net type: %s]...' % net_type)

    if 'embedding_config' in layers and layers['embedding_config']:
        input_class = Input(shape=(1, ), name='input_class')
        conf = dict(layers['embedding_config'], **{'input_dim': nb_classes})
        x = Embedding.from_config(conf)(input_class)
        if 'dropout' in layers and layers['dropout']:
            x = SpatialDropout1D(rate=layers['dropout'])(x)
    else:
        input_class = Input(shape=(nb_classes, ), name='input_class')
        x = Dense(reduce(operator.mul, latent_shape))(input_class)
    x = Reshape(latent_shape)(x)

    input_latent = Input(shape=latent_shape, name='input_latent')
    x = multiply([x, input_latent])

    if net_type == 'cnn':
        for i in range(blocks):
            if 'bn_config' in layers and layers['bn_config']:
                x = cnn.upblock2d_bn(x, transpose=i != blocks - 1, **layers)
            else:
                x = cnn.upblock2d(x, transpose=i != blocks - 1, **layers)
    elif net_type == 'densely':
        filters = latent_shape[0 if K.image_data_format() == "channels_first"
                               else -1]
        for i in range(blocks):
            x, filters = densely.upblock2d(x, filters,
                                           transpose=i != blocks - 1, **layers)
    img = Conv2D(3, (1, 1), activation='tanh', padding='same')(x)

    # Model
    generator = Model(inputs=[input_class, input_latent], outputs=img,
                      name='generator')
    generator.compile(loss=loss, optimizer=deserialize(optimizer),
                      metrics=metrics)
    return generator


@ingredients.capture(prefix='discriminator')
def build_discriminator(nb_classes, input_shape, blocks, net_type, layers,
                        loss, optimizer, metrics):
    assert net_type in ['cnn', 'densely']

    print('Building discriminator [net type: %s]...' % net_type)

    input_image = Input(shape=input_shape, name='input_image')
    if net_type == 'cnn':
        for i in range(blocks):
            if 'bn_config' in layers and layers['bn_config']:
                x = cnn.block2d_bn(input_image if i == 0 else x,
                                   pool=i != blocks - 1, **layers)
            else:
                x = cnn.block2d(input_image if i == 0 else x,
                                pool=i != blocks - 1, **layers)
    elif net_type == 'densely':
        filters = input_shape[0 if K.image_data_format() == 'channels_first'
                              else 2]
        for i in range(blocks):
            if 'bn_config' in layers and layers['bn_config']:
                x, filters = densely.block2d_bn(input_image if i == 0 else x,
                                                filters, pool=i != blocks - 1,
                                                **layers)
            else:
                x, filters = densely.block2d(input_image if i == 0 else x,
                                             filters, pool=i != blocks - 1,
                                             **layers)

    # fake
    f = Conv2D(1, (4, 4))(x)
    f = Flatten()(f)
    f = Activation(layers['f_activation'], name='f')(f)

    # prediction
    p = Conv2D(nb_classes, (4, 4))(x)
    p = Flatten()(p)
    p = Activation('softmax', name='p')(p)

    # Model
    discriminator = Model(inputs=input_image, outputs=[f, p],
                          name='discriminator')
    discriminator.compile(loss=loss, optimizer=deserialize(optimizer),
                          metrics=metrics)
    return discriminator


@ingredients.capture(prefix='combined')
def build_combined(generator, discriminator, loss, optimizer, metrics):
    print('Building combined model...')

    class_shape = generator.get_layer('input_class').input_shape[1:]
    image_class = Input(shape=class_shape, name='combined_input_class')

    latent_shape = generator.get_layer('input_latent').input_shape[1:]
    latent = Input(shape=latent_shape, name='combined_input_latent')
    fake = generator([image_class, latent])

    discriminator.trainable = False
    fake, prediction = discriminator(fake)
    combined = Model(inputs=[image_class, latent], outputs=[fake, prediction])
    combined.compile(loss=loss, optimizer=deserialize(optimizer),
                     metrics=metrics)
    return combined
