# -*- coding: utf-8 -*-

from copy import deepcopy
from keras import backend as K
from keras.layers import (deserialize as deserialize_layer, Dense, Embedding,
                          Input, SpatialDropout1D)
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers
from ingredients.models import ingredients


@ingredients.capture
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
        x = SpatialDropout1D(rate=layers['embedding_dropout'])(x)

    for i in range(N):
        layer = deepcopy(layers['recurrent_config'])
        if i != N - 1:
            layer['config']['return_sequences'] = True
        x = deserialize_layer(layer)(x)

    # outputs
    output_types = ['class', 'vec']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = []
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics.append(output['metrics'])

        if output['t'] == 'class':
            layer = deepcopy(layers[output['layer']])
            layer['config']['units'] = output['nb_classes']
            if 'activation' in output:
                layer['config']['activation'] = output['activation']
            if 'name' in output:
                layer['config']['name'] = output['name']
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
