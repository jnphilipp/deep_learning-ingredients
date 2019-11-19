# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import deserialize as deserialize_layer

from . import ingredient


@ingredient.capture
def inputs(inputs: list, layers: dict, *args, **kwargs):
    model_inputs = []
    xs = []
    tensors: dict = {}
    for _input in inputs:
        if _input['t'] == 'embedding':
            x = Input(shape=_input['shape'], name=_input['name'])
            model_inputs.append(x)

            if 'Embedding' not in tensors:
                tensors['Embedding'] = \
                    Embedding.from_config(layers['embedding'])
            x = tensors['Embedding'](x)
            if layers['embedding_dropout']:
                if layers['embedding_dropout']['class_name'] not in tensors:
                    tensors[layers['embedding_dropout']['class_name']] = \
                        deserialize_layer(layers['embedding_dropout'])
                x = tensors[layers['embedding_dropout']['class_name']](x)
            xs.append(x)
        elif _input['t'] == 'input':
            x = Input(shape=_input['shape'], name=_input['name'])
            model_inputs.append(x)
            xs.append(x)

    return model_inputs, xs
