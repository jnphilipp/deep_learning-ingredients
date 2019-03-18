# -*- coding: utf-8 -*-

from copy import deepcopy
from keras.layers import *
from keras.layers import deserialize as deserialize_layer

from . import ingredient


@ingredient.capture
def outputs(vec, layers, outputs, *args, **kwargs):
    output_types = ['class', 'image', 'mask', 'vec']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    if 'batchnorm' in layers:
        batchnorm = layers['batchnorm']
        if 'bottleneck2d' in layers:
            bottleneck_activation = layers['bottleneck2d']['activation']
            bottleneck2d = dict(layers['bottleneck2d'],
                                **{'activation': 'linear'})
        else:
            bottleneck2d = None
    else:
        batchnorm = None
        if 'bottleneck2d' in layers:
            bottleneck2d = layers['bottleneck2d']
        else:
            bottleneck2d = None
    dense = layers['dense'] if 'dense' in layers else {}
    conv2d = layers['conv2d'] if 'conv2d' in layers else {}
    conv2dt = layers['conv2dt'] if 'conv2dt' in layers else {}

    outs = []
    loss = []
    metrics = {}
    for output in outputs:
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
                    'padding': 'valid'}))(vec)
                x = Flatten()(x)
                outs.append(Activation(activation, name=name)(x))
            elif output['layer'] == 'dense':
                outs.append(Dense.from_config(dict(dense, **{
                    'units': nb_classes,
                    'activation': activation,
                    'name': name}))(vec))
        elif output['t'] == 'image':
            outs.append(Conv2D.from_config(dict(conv2d, **{
                'filters': 1 if output['grayscale'] else 3,
                'kernel_size': (1, 1),
                'padding': 'same',
                'activation': activation,
                'name': name}))(vec))
        elif output['t'] == 'mask':
            shortcuts = kwargs['shortcuts']

            s = vec
            for i in reversed(range(len(shortcuts))):
                shortcut = shortcuts[i][0]
                filters = shortcuts[i - 1 if i >= 0 else 0][1]
                if i is not len(shortcuts) - 1:
                    s = concatenate([s, shortcut], axis=layers['concat_axis'])
                else:
                    s = shortcut
                if i > 0:
                    if bottleneck2d is not None:
                        s = Conv2D.from_config(dict(bottleneck2d,
                                                    **{'filters': filters}))(s)
                        if batchnorm is not None:
                            s = BatchNormalization.from_config(batchnorm)(s)
                            s = Activation(bottleneck_activation)(s)
                    s = Conv2DTranspose.from_config(dict(conv2dt, **{
                        'filters': filters}))(s)

            outs.append(Conv2D.from_config(dict(conv2d, **{
                'filters': nb_classes,
                'name': name,
                'activation': activation}))(s))
        elif output['t'] == 'vec':
            outs.append(vec)

    return outs, loss, metrics
