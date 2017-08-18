# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import (deserialize as deserialize_layers, Conv1D, Embedding,
                          Input, RepeatVector, SpatialDropout1D)
from keras.models import Model
from keras.optimizers import deserialize as deserialize_optimizers
from ingredients.models import ingredients


@ingredients.config
def config():
    seq2seq = {
        'layers': {
            'embedding_config': {
                'output_dim': 128,
                'mask_zero': True,
                'name': 'embedding'
            },
            'embedding_dropout': 0.1,
            'recurrent_in_config': {
                'class_name': 'GRU',
                'config': {
                    'units': 512,
                    'dropout': 0.1,
                    'recurrent_dropout': 0.1,
                    'name': 'recurrent_in'
                }
            },
            'repeatvector_config': {
                'name': 'repeatvector'
            },
            'recurrent_out_config': {
                'class_name': 'GRU',
                'config': {
                    'units': 512,
                    'dropout': 0.1,
                    'recurrent_dropout': 0.1,
                    'return_sequences': True,
                    'name': 'recurrent_out'
                }
            },
            'conv1d_config': {
                'kernel_size': 1,
                'activation': 'softmax',
                'name': 'output'
            }
        },
        'loss': 'categorical_crossentropy',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy'],
        'sample_weight_mode': 'temporal'
    }


@ingredients.capture(prefix='seq2seq')
def build(vocab_size, max_len, layers, loss, optimizer, metrics,
          sample_weight_mode):
    print('Building Seq2Seq...')

    inputs = Input(shape=(None,), name='input')
    x = Embedding.from_config(dict(layers['embedding_config'],
                                   **{'input_dim': vocab_size}))(inputs)
    if layers['embedding_dropout']:
        x = SpatialDropout1D(rate=layers['embedding_dropout'])(x)

    x = deserialize_layers(layers['recurrent_in_config'])(x)
    x = RepeatVector.from_config(dict(layers['repeatvector_config'],
                                      **{'n': max_len}))(x)
    x = deserialize_layers(layers['recurrent_out_config'])(x)
    output = Conv1D.from_config(dict(layers['conv1d_config'],
                                     **{'filters': vocab_size}))(x)

    # Model
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=deserialize_optimizers(optimizer),
                  metrics=metrics, sample_weight_mode=sample_weight_mode)
    return model
