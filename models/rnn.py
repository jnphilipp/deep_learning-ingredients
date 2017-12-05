# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import (deserialize as deserialize_layers, Dense, Embedding,
                          Input, SpatialDropout1D)
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers
from ingredients.models import ingredients


@ingredients.config
def config():
    rnn = {
        'layers': {
            'embedding_config': {
                'output_dim': 128,
                'mask_zero': True,
                'name': 'embedding'
            },
            'embedding_dropout': 0.1,
            'recurrent_config': {
                'class_name': 'GRU',
                'config': {
                    'units': 512,
                    'dropout': 0.1,
                    'recurrent_dropout': 0.1,
                    'name': 'gru'
                }
            },
            'dense_config': {
                'activation': 'softmax',
                'name': 'output'
            }
        },
        'loss': 'categorical_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy']
    }


@ingredients.capture(prefix='rnn')
def build(vocab_size, nb_classes, layers, loss, optimizer, metrics):
    print('Building RNN...')

    inputs = Input(shape=(None,), name='input')
    x = Embedding.from_config(dict(layers['embedding_config'],
                                   **{'input_dim': vocab_size}))(inputs)
    if layers['embedding_dropout']:
        x = SpatialDropout1D(rate=layers['embedding_dropout'])(x)

    vec = deserialize_layers(layers['recurrent_config'])(x)
    if type(nb_classes) == int:
        output = Dense.from_config(dict(layers['dense_config'],
                                    **{'units': nb_classes}))(x)
    else:
        output = []
        for i, units in enumerate(nb_classes):
            dense_config = layers['dense_config'].copy()
            if 'name' in dense_config:
                dense_config['name'] += str(i)
            output.append(Dense.from_config(dict(dense_config,
                                                 **{'units': units}))(vec))

    # Model
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics)
    return model