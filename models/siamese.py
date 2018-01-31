# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Activation, Input, Lambda
from keras.models import Model
from keras.optimizers import deserialize
from ingredients.models import cnn, ingredients


@ingredients.config
def config():
    siamese = {
        'inner_net_type': 'cnn',
        'loss': 'mse',
        'optimizer': {
            'class_name': 'adam',
            'config': {}
        },
        'metrics': ['accuracy']
    }


@ingredients.capture(prefix='siamese')
def build(inner_net_type, loss, optimizer, metrics, output_names=['distance'],
          *args, **kwargs):
    assert inner_net_type in ['cnn']

    print('Building Siamese [inner net type: %s]...' % inner_net_type)
    if inner_net_type == 'cnn':
        inner_model = cnn.build(**kwargs)

    input_r = Input(inner_model.get_layer('input').input_shape[1:],
                    name='input_r')
    input_l = Input(inner_model.get_layer('input').input_shape[1:],
                    name='input_l')

    xr = inner_model(input_r)
    xl = inner_model(input_l)

    outputs = [Lambda(lambda x: K.mean(K.abs(x[0] - x[1]), axis=-1),
                      name=output_names[i], output_shape=(1,))([xr, xl])
               for i in range(len(output_names))]

    siamese_model = Model(inputs=[input_r, input_l], outputs=outputs,
                          name=kwargs['name'] if 'name' in kwargs else 'siamese')
    siamese_model.compile(loss=loss, optimizer=deserialize(optimizer),
                          metrics=metrics)
    return siamese_model
