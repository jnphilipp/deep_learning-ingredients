# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020
#               J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of deep_learning-ingredients.
#
# deep_learning-ingredients is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# deep_learning-ingredients is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with deep_learning-ingredients. If not, see
# <http://www.gnu.org/licenses/>.

from logging import Logger
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Tuple, Union

from . import ingredient
from .. import models


@ingredient.capture
def build(encoder_net_type: str, decoder_net_type: str, loss: Union[list, str],
          metrics: Union[dict, str], optimizer: Optimizer, _log: Logger,
          loss_weights: Optional[Union[List, Dict]] = None,
          sample_weight_mode: Optional[Union[str, Dict[str, str],
                                             List[str]]] = None,
          weighted_metrics: Optional[List] = None,
          target_tensors: Optional[Union[Tensor, List[Tensor]]] = None, *args,
          **kwargs) -> Tuple[Model, Model, Model]:
    if 'name' in kwargs:
        name = kwargs.pop('name')
    else:
        name = 'autoencoder'
    _log.info(f'Build AutoEncoder model [{name}]')

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
