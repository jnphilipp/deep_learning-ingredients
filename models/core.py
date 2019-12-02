# -*- coding: utf-8 -*-
# Copyright (C) 2019 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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

import os
import sys

from ingredients import callbacks, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.layer_utils import count_params

from . import autoencoder, cnn, dense, ingredient, rnn, siamese


@ingredient.config
def config():
    path = None
    inputs = []
    layers = {}
    merge = {}
    outputs = []


@ingredient.capture
def get(path, net_type, _log, *args, **kwargs):
    net_types = ['autoencoder', 'cnn', 'dense', 'rnn', 'siamese']
    assert net_type in net_types

    if 'callbacks' in kwargs:
        return_callbacks = kwargs.pop('callbacks')
    else:
        return_callbacks = True

    kwargs['optimizer'] = optimizers.get()
    if not path or not os.path.exists(path):
        if net_type == 'autoencoder':
            model = autoencoder.build(*args, **kwargs)
        elif net_type == 'cnn':
            model = cnn.build(*args, **kwargs)
        elif net_type == 'dense':
            model = dense.build(*args, **kwargs)
        elif net_type == 'rnn':
            model = rnn.build(*args, **kwargs)
        elif net_type == 'siamese':
            model = siamese.build(*args, **kwargs)
    else:
        model = load()

    model._check_trainable_weights_consistency()
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)
    non_trainable_count = count_params(model.non_trainable_weights)

    _log.info(f'Total params: {trainable_count + non_trainable_count:,}')
    _log.info(f'Trainable params: {trainable_count:,}')
    _log.info(f'Non-trainable params: {non_trainable_count:,}')

    if return_callbacks:
        if 'callbacks_config' in kwargs:
            return model, callbacks.get(**kwargs['callbacks_config'])
        else:
            return model, callbacks.get()
    else:
        return model


@ingredient.capture
def load(path, _log):
    if path is None:
        _log.critical('No path given to load model.')
    elif isinstance(path, str):
        _log.info(f'Load model [{path}]')
        return load_model(path)
    else:
        models = []
        _log.info(f'Load model [{", ".join(path)}]')
        for p in path:
            models.append(load_model(p))
        return models


@ingredient.capture
def save(model, name='model', _log=None, _run=None):
    _log.info(f'Save model [{name}]')

    path = os.path.join(_run.observers[0].run_dir, f'{name}.json')
    with open(path, 'w', encoding='utf8') as f:
        f.write(model.to_json())
        f.write('\n')

    path = os.path.join(_run.observers[0].run_dir, f'{name}_summary')
    stdout = sys.stdout
    with open(path, 'w', encoding='utf8') as f:
        sys.stdout = f
        model.summary()
    sys.stdout = stdout

    model.save(os.path.join(_run.observers[0].run_dir, f'{name}.h5'))


@ingredient.capture
def make_function(model, input_layers, output_layers, _log):
    _log.info('Make function')

    def get_layer(config):
        if 'name' in layer:
            return model.get_layer(layer['name'])
        elif 'idx' in layer:
            return model.get_layer(index=layer['idx'])
        elif 'index' in layer:
            return model.get_layer(index=layer['index'])
        else:
            return None

    inputs = []
    for layer in input_layers:
        if 'at' in layer:
            inputs.append(get_layer(layer).get_input_at(layer['at']))
        elif 'node_idx' in layer:
            inputs.append(get_layer(layer).get_input_at(layer['node_idx']))
        elif 'node_index' in layer:
            inputs.append(get_layer(layer).get_input_at(layer['node_index']))
        else:
            inputs.append(get_layer(layer).input)
    inputs.append(K.learning_phase())

    outputs = []
    for layer in output_layers:
        if 'at' in layer:
            outputs.append(get_layer(layer).get_output_at(layer['at']))
        elif 'node_idx' in layer:
            outputs.append(get_layer(layer).get_output_at(layer['node_idx']))
        elif 'node_index' in layer:
            outputs.append(get_layer(layer).get_output_at(layer['node_index']))
        else:
            outputs.append(get_layer(layer).output)
    return K.function(inputs, outputs)


@ingredient.command
def summary(model=None, _log=None):
    _log.info(f'Model summary')
    if model is None:
        model = get(callbacks=False)
    model.summary()


@ingredient.command
def plot(model=None, name='model', _log=None, _run=None):
    _log.info(f'Plot {name}')
    if model is None:
        model = get(callbacks=False)
    model.summary()
    plot_model(model, to_file=os.path.join(_run.observers[0].run_dir,
                                           f'{name}.png'))
