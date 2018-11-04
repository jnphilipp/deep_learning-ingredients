# -*- coding: utf-8 -*-

from copy import deepcopy
from keras import backend as K
from keras.layers import *
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers

from . import ingredient


@ingredient.capture
def build(vocab_size, N, layers, outputs, optimizer, _log, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          *args, **kwargs):
    if 'name' in kwargs:
        _log.info('Build RNN model [%s]' % kwargs['name'])
    else:
        _log.info('Build RNN model')

    inputs = Input(shape=(None,), name='input')
    x = Embedding.from_config(dict(layers['embedding_config'],
                                   **{'input_dim': vocab_size}))(inputs)
    if layers['embedding_dropout']:
        x = deserialize_layer(layers['embedding_dropout'])(x)

    for i in range(N):
        layer = deepcopy(layers['recurrent_config'])
        if i != N - 1:
            layer['config']['return_sequences'] = True

        if 'bidirectional_config' in layers and layers['bidirectional_config']:
            conf = dict(layers['bidirectional_config'],
                        **{'layer': layer})
            x = Bidirectional.from_config(conf)(x)
        else:
            x = deserialize_layer(layer)(x)

    # outputs
    output_types = ['class', 'vec']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = {}
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics[output['name']] = output['metrics']

        if output['t'] == 'class':
            nb_classes = output['nb_classes']
            layer = deepcopy(layers[output['layer']])
            layer['config'][nb_classes['k']] = nb_classes['v']
            layer['config']['name'] = output['name']
            if 'activation' in output:
                layer['config']['activation'] = output['activation']
            outs.append(deserialize_layer(layer)(x))
        elif output['t'] == 'vec':
            outs.append(x)

    # Model
    model = Model(inputs=inputs, outputs=outs,
                  name=kwargs['name'] if 'name' in kwargs else 'rnn')
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics, loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
