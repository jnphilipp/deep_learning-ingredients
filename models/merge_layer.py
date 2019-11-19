# -*- coding: utf-8 -*-

from tensorflow.keras.layers import (add, average, concatenate, dot, maximum,
                                     minimum, multiply, subtract)
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.framework.ops import Tensor
from typing import Iterable

from . import ingredient


@ingredient.capture(prefix='merge')
def merge_layer(inputs: Iterable[Tensor], t: str, config: dict = {}):
    if t == 'add':
        return add(inputs, **config)
    elif t == 'average':
        return average(inputs, **config)
    elif t == 'concatenate':
        return concatenate(inputs, **config)
    elif t == 'dot':
        return dot(inputs, **config)
    elif t == 'maximum':
        return maximum(inputs, **config)
    elif t == 'minimum':
        return minimum(inputs, **config)
    elif t == 'multiply':
        return multiply(inputs, **config)
    elif t == 'subtract':
        return subtract(inputs, **config)
