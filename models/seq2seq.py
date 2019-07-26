# -*- coding: utf-8 -*-

from copy import deepcopy
from keras import backend as K
from keras.layers import *
from keras.layers import deserialize as deserialize_layer
from keras.models import Model

from . import ingredient


@ingredient.capture
def build(vocab_size, N, layers, outputs, optimizer, _log,
          loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
          target_tensors=None, *args, **kwargs):
    if 'name' in kwargs:
        _log.info('Build Seq2Seq model [%s]' % kwargs['name'])
    else:
        _log.info('Build Seq2Seq model')

    inputs = Input(shape=(None,), name='input')
    x = Embedding.from_config(dict(layers['embedding_config'],
                                   **{'input_dim': vocab_size}))(inputs)
    if 'embedding_dropout' in layers and layers['embedding_dropout']:
        x = SpatialDropout1D(rate=layers['embedding_dropout'])(x)

    if type(N) == int:
        N = (N,) * (len(outputs) + 1)
    assert len(N) == len(outputs) + 1

    for i in range(N[0]):
        layer = deepcopy(layers['recurrent_in_config'])
        if i != N[0] - 1:
            layer['config']['return_sequences'] = True

        if 'bidirectional_config' in layers and layers['bidirectional_config']:
            conf = dict(layers['bidirectional_config'],
                        **{'layer': layer})
            x = Bidirectional.from_config(conf)(x)
        else:
            x = deserialize_layer(layer)(x)
    vec = x

    # outputs
    output_types = ['seq']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = {}

    conv1d_config = dict(layers['conv1d_config'],
                         **{'filters': vocab_size})
    if 'repeatvector_config' not in layers:
        layers['repeatvector_config'] = {}
    for i, output in enumerate(outputs):
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics[output['name']] = output['metrics']

        conv1d_config['name'] = output['name']
        if output['t'] == 'seq':
            x = RepeatVector.from_config(dict(layers['repeatvector_config'],
                                              **{'n': output['max_len']}))(vec)

            for j in range(N[i + 1]):
                layer = deepcopy(layers['recurrent_out_config'])
                layer['config']['return_sequences'] = True
                x = deserialize_layer(layer)(x)
            outs.append(Conv1D.from_config(conv1d_config)(x))

    # Model
    model = Model(inputs=inputs, outputs=outs,
                  name=kwargs['name'] if 'name' in kwargs else 'seq2seq')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode,
                  weighted_metrics=weighted_metrics,
                  target_tensors=target_tensors)
    return model
