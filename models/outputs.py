# -*- coding: utf-8 -*-

from keras.layers import *
from keras.layers import deserialize as deserialize_layer

from . import ingredient


@ingredient.capture
def outputs(vec, blocks, layers, outputs, *args, **kwargs):
    output_types = ['class', 'image', 'mask', 'vec']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = {}
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics[output['name']] = output['metrics']

        if output['t'] == 'class':
            if 'layer' in output:
                layer = output['layer']
            elif 'conv2d_config' in layers:
                layer = 'conv2d'
            elif 'dense_config' in layers:
                layer = 'dense'

            if layer == 'conv2d':
                x = Conv2D.from_config(dict(layers['conv2d_config'],
                                       **{'filters': output['nb_classes'],
                                          'kernel_size': (kwargs['rows'],
                                                          kwargs['cols']),
                                          'padding': 'valid'}))(vec)
                x = Flatten()(x)
                outs.append(Activation(output['activation'],
                                       name=output['name'])(x))
            elif layer == 'dense':
                conf = dict(layers['dense_config'],
                            **{'units': output['nb_classes'],
                               'activation': output['activation'],
                               'name': output['name']})
                outs.append(Dense.from_config(conf)(x))
        elif output['t'] == 'image':
            conf = dict(layers['conv2d_config'],
                        **{'filters': 1 if output['grayscale'] else 3,
                           'kernel_size': (1, 1),
                           'padding': 'same',
                           'activation': output['activation'],
                           'name': output['name']})
            outs.append(Conv2D.from_config(conf)(vec))
        elif output['t'] == 'mask':
            bottleneck2d_config = layers['bottleneck2d_config']
            bn_config = layers['bn_config'] if 'bn_config' in layers else None
            conv2d_config = layers['conv2d_config']
            shortcuts = kwargs['shortcuts']

            s = vec
            for i in reversed(range(blocks)):
                shortcut = shortcuts[i][0]
                filters = shortcuts[i - 1 if i >= 0 else 0][1]
                if i is not blocks - 1:
                    s = concatenate([s, shortcut], axis=layers['concat_axis'])
                else:
                    s = shortcut
                if i > 0:
                    if layers['bottleneck']:
                        conf = dict(bottleneck2d_config,
                                    **{'filters': filters})
                        s = Conv2D.from_config(conf)(s)
                        if bn_config:
                            s = BatchNormalization.from_config(bn_config)(s)
                            s = Activation(layers['activation'])(s)
                    conf = dict(conv2d_config, **{'strides': layers['strides'],
                                                  'filters': filters})
                    s = Conv2DTranspose.from_config(conf)(s)

            filters = output['nb_classes'] if 'nb_classes' in output else 1
            activation = output['activation'] if 'activation' in output \
                else 'sigmoid'
            conf = dict(conv2d_config, **{'filters': filters,
                                          'name': output['name'],
                                          'activation': activation})
            outs.append(Conv2D.from_config(conf)(s))
        elif output['t'] == 'vec':
            outs.append(vec)

    return outs, loss, metrics
