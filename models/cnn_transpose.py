# -*- coding: utf-8 -*-

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from . import ingredient


@ingredient.capture
def build(input_shape, outputs, optimizer, loss_weights=None,
          sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
          _log=None, *args, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
            _log.info(f'Build CNN transpose model [{name}]')
        else:
            name = 'cnn-transpose'
            _log.info('Build CNN transpose model')

        if 'embedding_config' in layers and layers['embedding_config']:
            input_class = Input(shape=(1, ), name='input_class')
            conf = dict(layers['embedding_config'], **{'input_dim': nb_classes})
            x = Embedding.from_config(conf)(input_class)
            if 'dropout' in layers and layers['dropout']:
                x = SpatialDropout1D(rate=layers['dropout'])(x)
        else:
            input_class = Input(shape=(nb_classes, ), name='input_class')
            x = Dense(reduce(operator.mul, latent_shape))(input_class)
        x = Reshape(latent_shape)(x)

        input_latent = Input(shape=latent_shape, name='input_latent')
        x = multiply([x, input_latent])


        latent_shape = (8, 4, 4)
        inputs = [
            {
                't': 'embedding',
                'input_dim': nb_classes,
                'shape': (1,),
                'name': 'class'
            }, {
                't': 'vec',
                'input_shape': (nb_classes,),
                'name': 'class_vec'
            }, {
                't': 'latent',
                'name': 'latent'
            }
        ]

        for i in inputs:
            if i['t'] == 'embedding':
                input_class = Input(shape=i['shape'], name=i['name'])
                conf = dict(layers['embedding_config'], **{'input_dim': i['input_dim']})
                x = Embedding.from_config(conf)(input_class)
                if 'dropout' in layers and layers['dropout']:
                    x = SpatialDropout1D(rate=layers['dropout'])(x)
