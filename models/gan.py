# -*- coding: utf-8 -*-

import operator

from functools import reduce
from keras import backend as K
from keras.layers import (multiply, Activation, Conv2D, Dense, Embedding,
                          Flatten, Input, Reshape, SpatialDropout1D)
from keras.models import Model
from keras.optimizers import deserialize

from . import ingredient


@ingredient.capture
def build(generator_net_type, discriminator_net_type, optimizer,
          loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
          target_tensors=None,, _log=None, *args, **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
        del kwargs['name']
        _log.info('Build GAN [%s/%s] model [%s]' % (generator_net_type,
                                                    discriminator_net_type,
                                                    name))
    else:
        name = 'gan'
        _log.info('Build GAN [%s/%s] model' % (generator_net_type,
                                               discriminator_net_type))

    generator = models.get(None, generator_net_type,
                           outputs=[{'t': 'vec', 'loss': 'mse'}],
                           name='generator', *args, **kwargs)

    # class_shape = generator.get_layer('input_class').input_shape[1:]
    # image_class = Input(shape=class_shape, name='combined_input_class')

    # latent_shape = generator.get_layer('input_latent').input_shape[1:]
    # latent = Input(shape=latent_shape, name='combined_input_latent')
    # fake = generator([image_class, latent])

    # discriminator.trainable = False
    # fake, prediction = discriminator(fake)
    # combined = Model(inputs=[image_class, latent], outputs=[fake, prediction],
    #                  name='gan')
    # combined.compile(loss=loss, optimizer=deserialize(optimizer),
    #                  metrics=metrics)

    discriminator = None
    combined = None

    return generator, discriminator, combined


# @ingredients.capture(prefix='generator')
# def build_generator(nb_classes, latent_shape, blocks, net_type, layers, loss,
#                     optimizer, metrics):
#     assert net_type in ['cnn', 'densely']

#     print('Building generator [net type: %s]...' % net_type)

#     if 'embedding_config' in layers and layers['embedding_config']:
#         input_class = Input(shape=(1, ), name='input_class')
#         conf = dict(layers['embedding_config'], **{'input_dim': nb_classes})
#         x = Embedding.from_config(conf)(input_class)
#         if 'dropout' in layers and layers['dropout']:
#             x = SpatialDropout1D(rate=layers['dropout'])(x)
#     else:
#         input_class = Input(shape=(nb_classes, ), name='input_class')
#         x = Dense(reduce(operator.mul, latent_shape))(input_class)
#     x = Reshape(latent_shape)(x)

#     input_latent = Input(shape=latent_shape, name='input_latent')
#     x = multiply([x, input_latent])

#     if net_type == 'cnn':
#         for i in range(blocks):
#             if 'bn_config' in layers and layers['bn_config']:
#                 x = cnn.upblock2d_bn(x, transpose=i != blocks - 1, **layers)
#             else:
#                 x = cnn.upblock2d(x, transpose=i != blocks - 1, **layers)
#     elif net_type == 'densely':
#         filters = latent_shape[0 if K.image_data_format() == "channels_first"
#                                else -1]
#         for i in range(blocks):
#             x, filters = densely.upblock2d(x, filters,
#                                            transpose=i != blocks - 1, **layers)
#     img = Conv2D(3, (1, 1), activation='tanh', padding='same')(x)

#     # Model
#     generator = Model(inputs=[input_class, input_latent], outputs=img,
#                       name='generator')
#     generator.compile(loss=loss, optimizer=deserialize(optimizer),
#                       metrics=metrics)
#     return generator


# @ingredients.capture(prefix='discriminator')
# def build_discriminator(grayscale, rows, cols, blocks, nb_classes, net_type,
#                         layers, loss, optimizer, metrics):
#     assert net_type in ['cnn', 'densely']

#     print('Building discriminator [net type: %s]...' % net_type)

#     filters = 1 if grayscale else 3
#     if K.image_data_format() == 'channels_first':
#         input_shape = (filters, rows, cols)
#     else:
#         input_shape = (rows, cols, filters)

#     inputs = Input(shape=input_shape, name='input')
#     if net_type == 'cnn':
#         for i in range(blocks):
#             if 'bn_config' in layers and layers['bn_config']:
#                 x = cnn.block2d_bn(inputs if i == 0 else x,
#                                    pool=i != blocks - 1, **layers)
#             else:
#                 x = cnn.block2d(inputs if i == 0 else x,
#                                 pool=i != blocks - 1, **layers)
#     elif net_type == 'densely':
#         for i in range(blocks):
#             if 'bn_config' in layers and layers['bn_config']:
#                 x, filters = densely.block2d_bn(inputs if i == 0 else x,
#                                                 filters, pool=i != blocks - 1,
#                                                 **layers)
#             else:
#                 x, filters = densely.block2d(inputs if i == 0 else x,
#                                              filters, pool=i != blocks - 1,
#                                              **layers)

#     # fake
#     f = Conv2D(1, (4, 4))(x)
#     f = Flatten()(f)
#     f = Activation(layers['f_activation'], name='f')(f)

#     # prediction
#     p = Conv2D(nb_classes, (4, 4))(x)
#     p = Flatten()(p)
#     p = Activation('softmax', name='p')(p)

#     # Model
#     discriminator = Model(inputs=inputs, outputs=[f, p],
#                           name='discriminator')
#     discriminator.compile(loss=loss, optimizer=deserialize(optimizer),
#                           metrics=metrics)
#     return discriminator
