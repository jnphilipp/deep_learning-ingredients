# -*- coding: utf-8 -*-

from copy import deepcopy
from keras import backend as K
from keras.layers import *
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers

from . import ingredient
from .outputs import outputs


@ingredient.capture
def build(vocab_size, N, layers, optimizer, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          _log=None, *args, **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
        _log.info(f'Build RNN model [{name}]')
    else:
        name = 'rnn'
        _log.info('Build RNN model')

    inputs = Input(shape=(None,), name='input')
    x = Embedding.from_config(dict(layers['embedding'],
                                   **{'input_dim': vocab_size}))(inputs)
    if layers['embedding_dropout']:
        x = deserialize_layer(layers['embedding_dropout'])(x)

    for i in range(N):
        rnn_layer = deepcopy(layers['recurrent'])
        if i != N - 1:
            rnn_layer['config']['return_sequences'] = True

        if 'bidirectional' in layers and layers['bidirectional']:
            conf = dict(layers['bidirectional'], **{'layer': rnn_layer})
            x = Bidirectional.from_config(conf)(x)
        else:
            x = deserialize_layer(rnn_layer)(x)

    # outputs
    outs, loss, metrics = outputs(x)

    # Model
    model = Model(inputs=inputs, outputs=outs, name=name)
    model.compile(loss=loss, metrics=metrics, loss_weights=loss_weights,
                  optimizer=deserialize_optimizers(optimizer.copy()),
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
