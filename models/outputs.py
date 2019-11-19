# -*- coding: utf-8 -*-

from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import (Activation, BatchNormalization as BN,
                                     Bidirectional, Conv1D, Conv2D,
                                     Conv2DTranspose as Conv2DT, Dense,
                                     Flatten, RepeatVector)
from tensorflow.keras.layers import deserialize as deserialize_layer
from tensorflow.python.framework.ops import Tensor
from typing import Iterable, List, Optional, Union

from . import ingredient


@ingredient.capture
def outputs(vec: Union[Tensor, Iterable[Tensor]], layers: dict,
            outputs: List[dict], *args, **kwargs):
    output_types = ['class', 'image', 'mask', 'seq', 'vec']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    if 'batchnorm' in layers:
        batchnorm = layers['batchnorm']
        if 'bottleneck2d' in layers:
            bottleneck_activation = layers['bottleneck2d']['activation']
            bottleneck2d: Optional[dict] = dict(layers['bottleneck2d'],
                                                **{'activation': 'linear'})
        else:
            bottleneck2d = None
    else:
        batchnorm = None
        if 'bottleneck2d' in layers:
            bottleneck2d = layers['bottleneck2d']
        else:
            bottleneck2d = None
    concat_axis = layers['concat_axis'] if 'concat_axis' in layers else -1
    conv1d = layers['conv1d'] if 'conv1d' in layers else {}
    conv2d = layers['conv2d'] if 'conv2d' in layers else {}
    conv2dt = layers['conv2dt'] if 'conv2dt' in layers else {}
    dense = layers['dense'] if 'dense' in layers else {}
    rpvec = layers['repeatvector'] if 'repeatvector' in layers else {}

    if type(vec) == Tensor:
        vec = [vec]

    outs = []
    loss = []
    metrics = {}
    for output in outputs:
        for v in vec:
            activation = output['activation']
            name = output['name']
            nb_classes = output['nb_classes'] if 'nb_classes' in output else 1

            loss.append(output['loss'])
            if 'metrics' in output:
                metrics[output['name']] = output['metrics']

            if output['t'] == 'class':
                if output['layer'] == 'conv2d':
                    x = Conv2D.from_config(dict(conv2d, **{
                        'filters': nb_classes,
                        'kernel_size': (kwargs['rows'], kwargs['cols']),
                        'padding': 'valid'}))(v)
                    x = Flatten()(x)
                    outs.append(Activation(activation, name=name)(x))
                elif output['layer'] == 'dense':
                    outs.append(Dense.from_config(dict(dense, **{
                        'units': nb_classes,
                        'activation': activation,
                        'name': name}))(v))
            elif output['t'] == 'image':
                outs.append(Conv2D.from_config(dict(conv2d, **{
                    'filters': 1 if output['grayscale'] else 3,
                    'kernel_size': (1, 1),
                    'padding': 'same',
                    'activation': activation,
                    'name': name}))(v))
            elif output['t'] == 'mask':
                shortcuts = kwargs['shortcuts']

                s = v
                for i in reversed(range(len(shortcuts))):
                    shortcut = shortcuts[i][0]
                    filters = shortcuts[i - 1 if i >= 0 else 0][1]
                    if i is not len(shortcuts) - 1:
                        s = concatenate([s, shortcut], axis=concat_axis)
                    else:
                        s = shortcut
                    if i > 0:
                        if bottleneck2d is not None:
                            s = Conv2D.from_config(dict(bottleneck2d, **{
                                'filters': filters}))(s)
                            if batchnorm is not None:
                                s = BN.from_config(batchnorm)(s)
                                s = Activation(bottleneck_activation)(s)
                        s = Conv2DT.from_config(dict(conv2dt, **{
                            'filters': filters}))(s)

                outs.append(Conv2D.from_config(dict(conv2d, **{
                    'filters': nb_classes,
                    'name': name,
                    'activation': activation}))(s))
            elif output['t'] == 'seq':
                x = RepeatVector.from_config(dict(rpvec, **{
                    'n': output['max_len']}))(v)

                for j in range(output['N'] if 'N' in output else 1):
                    if 'recurrent' in output:
                        rnn_layer = dict(**layers[output['recurrent']])
                    elif 'bidirectional' in output:
                        rnn_layer = dict(**layers[output['bidirectional']])
                    else:
                        rnn_layer = dict(**layers[output['recurrent_out']])
                    rnn_layer['config'] = dict(rnn_layer['config'], **{
                            'return_sequences': True})

                    if 'bidirectional' in output:
                        conf = dict(layers['bidirectional'], **{
                            'layer': rnn_layer})
                        x = Bidirectional.from_config(conf)(x)
                    else:
                        x = deserialize_layer(rnn_layer)(x)
                outs.append(Conv1D.from_config(dict(conv1d, **{
                        'filters': nb_classes,
                        'activation': activation,
                        'name': name}))(x))
            elif output['t'] == 'vec':
                outs.append(v)

    return outs, loss, metrics
