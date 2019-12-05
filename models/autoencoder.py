# -*- coding: utf-8 -*-

from logging import Logger
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Optimizer
from typing import Tuple, Union

from . import ingredient
from .. import models


@ingredient.capture
def build(encoder_net_type: str, decoder_net_type: str, loss: Union[list, str],
          metrics: Union[dict, str], optimizer: Optimizer, _log: Logger,
          loss_weights: Union[list, dict] = None,
          sample_weight_mode: str = None, weighted_metrics: list = None,
          target_tensors=None, *args, **kwargs) -> Tuple[Model, Model, Model]:
    if 'name' in kwargs:
        name = kwargs['name']
        del kwargs['name']
        _log.info(f'Build AutoEncoder model [{name}]')
    else:
        name = 'autoencoder'
        _log.info('Build AutoEncoder model')

    encoder = models.get(None, encoder_net_type, name='encoder',
                         outputs=[{'t': 'vec', 'loss': 'mse'}], *args,
                         **kwargs)
    output_shape = encoder.get_layer('input').input_shape[1]
    decoder = models.get(None, decoder_net_type, name='decoder',
                         input_shape=encoder.output_shape[1:],
                         outputs=[{'t': 'vec', 'loss': 'mse'}],
                         output_shape=output_shape, *args, **kwargs)

    inputs = Input(shape=encoder.get_layer('input').input_shape[1:],
                   name='input')
    outputs = decoder(encoder(inputs))

    # Model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return encoder, decoder, model
