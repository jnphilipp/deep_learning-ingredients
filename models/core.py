# -*- coding: utf-8 -*-

import os
import sys

from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from ingredients.models import ingredients


@ingredients.config
def config():
    summary = False
    path = None


@ingredients.capture
def get(build_func, summary, path, *args, **kwargs):
    if build_func and not path:
        model = build_func(*args, **kwargs)
    else:
        model = load()
    if summary:
        model.summary()
    return model


@ingredients.capture
def load(path, _log):
    if path is None:
        _log.critical('No path given to load model.')
    elif isinstance(path, str):
        _log.info('Load model [%s]' % path)
        return load_model(path)
    else:
        models = []
        _log.info('Load model [%s]' % ', '.join(path))
        for p in path:
            models.append(load_model(p))
        return models


@ingredients.capture
def save(path, model, name=None):
    with open(os.path.join(path, '%s.json' % (name if name else 'model')),
              'w', encoding='utf8') as f:
        f.write(model.to_json())
        f.write('\n')

    stdout = sys.stdout
    with open(os.path.join(path, '%ssummary' % ('%s_' % name if name else '')),
              'w', encoding='utf8') as f:
        sys.stdout = f
        model.summary()
    sys.stdout = stdout

    model.save(os.path.join(path, '%s.h5' % (name if name else 'model')))


@ingredients.capture
def plot(model, path, name='model'):
    plot_model(model, to_file=os.path.join(path, '%s.png' % name))


@ingredients.capture
def make_function(model, input_layers, output_layers):
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
