# -*- coding: utf-8 -*-

from keras.layers import Input
from keras.models import Model
from keras.optimizers import deserialize

from . import ingredient
from .. import models


@ingredient.capture
def build(encoder_net_type, decoder_net_type, layers, loss, metrics, optimizer,
          _log, loss_weights=None, sample_weight_mode=None,
          weighted_metrics=None, target_tensors=None, *args, **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
        del kwargs['name']
        _log.info('Build AutoEncoder model [%s]' % kwargs['name'])
    else:
        name = 'autoencoder'
        _log.info('Build AutoEncoder model')

    encoder = models.get(None, encoder_net_type, name='encoder',
                         outputs=[{'t': 'vec', 'loss': 'mse'}], *args, **kwargs)
    decoder = models.get(None, decoder_net_type, name='decoder',
                         input_shape=encoder.output_shape[1:],
                         outputs=[{'t': 'vec', 'loss': 'mse'}],
                         output_shape=encoder.get_layer('input').input_shape[1],
                         *args, **kwargs)

    inputs = Input(shape=encoder.get_layer('input').input_shape[1:],
                   name='input')
    outputs = decoder(encoder(inputs))

    # Model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(loss=loss, optimizer=deserialize(optimizer), metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return encoder, decoder, model
