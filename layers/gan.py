# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import multiply, Activation, BatchNormalization, Conv2D, Embedding, Flatten, Input, Reshape, SpatialDropout1D
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.layers import densely, ingredients


@ingredients.config
def config():
    discriminator = {'bn_config': {'axis': 1},
                    'conv2d_config': {'kernel_size': (3, 3),
                                    'padding': 'same'
                    }
    }


@ingredients.capture(prefix='generator')
def build_generator(nb_classes, latent_shape, blocks, embedding_config, dropout, loss, optimizer, metrics):
    print('Building generator...')

    input_class = Input(shape=(1, ), name='input_class')
    x = Embedding.from_config({**embedding_config, **{'input_dim': nb_classes}})(input_class)
    if dropout:
        x = SpatialDropout1D(rate=dropout)(x)
    x = Reshape(latent_shape)(x)

    input_latent = Input(shape=latent_shape, name='input_latent')
    x = multiply([x, input_latent])

    filters = latent_shape[0 if K.image_data_format() == "channels_first" else -1]
    for i in range(blocks):
        x, filters = densely.upblock2d(x, filters, transpose=i != blocks - 1)
    img = Conv2D(3, (1, 1), activation='tanh', padding='same')(x)

    # Model
    generator = Model(inputs=[input_class, input_latent], outputs=img, name='generator')
    generator.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return generator


@ingredients.capture(prefix='discriminator')
def build_discriminator(nb_classes, input_shape, filters, blocks, bn_config, conv2d_config, activation, loss, optimizer, metrics):
    print('Building discriminator...')

    input_image = Input(shape=input_shape, name='input_image')
    if bn_config:
        x = Conv2D.from_config({**conv2d_config, **{'filters': filters}})(input_image)
        x = BatchNormalization.from_config(bn_config)(x)
        x = Activation(activation)(x)
    else:
        x = Conv2D.from_config({**conv2d_config, **{'filters': filters, 'activation': activation}})(input_image)

    for i in range(blocks):
        if bn_config:
            x, filters = densely.block2d_bn(x, filters, pool=i != blocks - 1)
        else:
            x, filters = densely.block2d(x, filters, pool=i != blocks - 1)

    # fake
    f = Conv2D(1, (4, 4))(x)
    f = Flatten()(f)
    f = Activation('sigmoid', name='f')(f)

    # prediction
    p = Conv2D(nb_classes, (4, 4))(x)
    p = Flatten()(p)
    p = Activation('softmax', name='p')(p)

    # Model
    discriminator = Model(inputs=input_image, outputs=[f, p], name='discriminator')
    discriminator.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return discriminator


@ingredients.capture(prefix='combined')
def build_combined(generator, discriminator, latent_shape, loss, optimizer, metrics):
    print('Building combined...')

    image_class = Input(shape=(1, ), name='combined_input_class')
    latent = Input(shape=latent_shape, name='combined_input_latent')
    fake = generator([image_class, latent])

    discriminator.trainable = False
    fake, prediction = discriminator(fake)
    combined = Model(inputs=[image_class, latent], outputs=[fake, prediction])
    combined.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics)
    return combined
