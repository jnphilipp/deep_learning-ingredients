# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.framework.ops import Tensor
from typing import List, Tuple

from . import ingredient


@ingredient.capture
def inputs(inputs: list, layers: dict, *args, **kwargs) -> \
        Tuple[List[Tensor], List[Tensor]]:
    model_inputs = []
    xs = []
    tensors: dict = {}
    for _input in inputs:
        if _input['t'] == 'embedding':
            x = Input(shape=_input['shape'], name=_input['name'])
            model_inputs.append(x)

            if 'embedding' not in tensors:
                tensors['embedding'] = \
                    Embedding.from_config(layers['embedding'])
            x = tensors['embedding'](x)
            if layers['embedding_dropout']:
                if layers['embedding_dropout']['class_name'] not in tensors:
                    tensors[layers['embedding_dropout']['class_name']] = \
                        deserialize_layer(layers['embedding_dropout'])
                x = tensors[layers['embedding_dropout']['class_name']](x)
            xs.append(x)
        elif _input['t'] == 'noise':
            x = Input(shape=_input['shape'], name=_input['name'])
            model_inputs.append(x)
            xs.append(deserialize_layer(layers['noise'])(x))
        elif _input['t'] == 'input':
            x = Input(shape=_input['shape'], name=_input['name'])
            model_inputs.append(x)
            xs.append(x)

    return model_inputs, xs
