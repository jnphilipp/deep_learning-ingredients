# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2019-2022 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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
from tensorflow.keras.layers import Input, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Sequence, Union

from . import core
from .ingredient import ingredient


@ingredient.capture
def build(
    generator_net: Dict,
    discriminator_net: Dict,
    loss: Union[str, List],
    metrics: Union[str, List, Dict],
    optimizer: Optimizer,
    _log: Logger,
    encoder_net: Optional[Dict] = None,
    loss_weights: Optional[Union[List, Dict]] = None,
    sample_weight_mode: Optional[Union[str, Dict[str, str], List[str]]] = None,
    weighted_metrics: Optional[List] = None,
    target_tensors: Optional[Union[Tensor, List[Tensor]]] = None,
    *args,
    **kwargs,
) -> Union[Sequence[Model]]:
    if "name" in kwargs:
        name = kwargs.pop("name")
    else:
        name = "gan"
    if encoder_net is None:
        _log.info(
            f'Build GAN [{generator_net["net_type"]}/'
            + f'{discriminator_net["net_type"]}] model [{name}]'
        )
    else:
        _log.info(
            f'Build GAN [({encoder_net["net_type"]},'
            + f'{generator_net["net_type"]})/'
            + f'{discriminator_net["net_type"]}] model [{name}]'
        )

    generator = core.get(None, name="generator", log_params=False, **generator_net)
    discriminator = core.get(
        None, name="discriminator", log_params=False, **discriminator_net
    )
    if encoder_net:
        encoder = core.get(None, name="encoder", log_params=False, **encoder_net)

    discriminator.trainable = False
    if encoder_net:
        generator_inputs = []
        for layer in generator.layers:
            if isinstance(layer, InputLayer):
                generator_inputs.append(Input(**layer.get_config()))
        generator_out = generator(generator_inputs)

        encoder_inputs = []
        for layer in encoder.layers:
            if isinstance(layer, InputLayer):
                encoder_inputs.append(Input(**layer.get_config()))
        encoder_out = encoder(encoder_inputs)

        outputs = discriminator([generator_out, encoder_out])

        combined = Model(
            inputs=generator_inputs + encoder_inputs, outputs=outputs, name=name
        )
        combined.compile(
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            optimizer=optimizer,
            sample_weight_mode=sample_weight_mode,
            weighted_metrics=weighted_metrics,
            target_tensors=target_tensors,
        )

        return generator, encoder, discriminator, combined
    else:
        inputs = []
        for layer in generator.layers:
            if isinstance(layer, InputLayer):
                inputs.append(Input(**layer.get_config()))
        generator_out = generator(inputs)

        outputs = discriminator(generator_out)

        combined = Model(inputs=inputs, outputs=outputs, name=name)
        combined.compile(
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            optimizer=optimizer,
            sample_weight_mode=sample_weight_mode,
            weighted_metrics=weighted_metrics,
            target_tensors=target_tensors,
        )
        return generator, discriminator, combined
