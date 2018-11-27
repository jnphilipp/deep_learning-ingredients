# -*- coding: utf-8 -*-

import math
import numpy as np
import os

from decorators import runtime
from ingredients.datasets.images import load_img
from keras import backend as K
from keras.callbacks import BaseLogger, CallbackList, History, ProgbarLogger
from keras.preprocessing.image import array_to_img

from . import ingredient


@ingredient.capture
def image(model, image_path, batch_size, overlap, rescale, data_format=None):
    def offset(size, diff, overlap):
        return math.floor(diff / math.ceil(diff / (size * (1 - overlap))))

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    if len(model.inputs) != 1:
        raise RuntimeError('Models with more than one input are not'
                           ' supported at the moment.')

    inputs = []
    for i in range(len(model.inputs)):
        name = model.inputs[i].name
        pos = min(name.index('/') if '/' in name else len(name),
                  name.index(':') if ':' in name else len(name))
        name = name[:pos]

        inputs.append({'shape': model.inputs[i].shape.as_list(), 'name': name})
        if data_format == 'channels_first':
            inputs[i]['grayscale'] = inputs[i]['shape'][1] == 1
            inputs[i]['r'] = inputs[i]['shape'][2]
            inputs[i]['c'] = inputs[i]['shape'][3]
        elif data_format == 'channels_last':
            inputs[i]['r'] = inputs[i]['shape'][1]
            inputs[i]['c'] = inputs[i]['shape'][2]
            inputs[i]['grayscale'] = inputs[i]['shape'][3] == 1

        inputs[i]['img'] = load_img(image_path, inputs[i]['grayscale'],
                                    rescale)
        if data_format == 'channels_first':
            inputs[i]['img_r'] = inputs[i]['img'].shape[1]
            inputs[i]['img_c'] = inputs[i]['img'].shape[2]
        elif data_format == 'channels_last':
            inputs[i]['img_r'] = inputs[i]['img'].shape[0]
            inputs[i]['img_c'] = inputs[i]['img'].shape[1]

        inputs[i]['diff_r'] = inputs[i]['img_r'] - inputs[i]['r']
        inputs[i]['diff_c'] = inputs[i]['img_c'] - inputs[i]['c']
        inputs[i]['offset_r'] = offset(inputs[i]['r'], inputs[i]['diff_r'],
                                       overlap)
        inputs[i]['offset_c'] = offset(inputs[i]['c'], inputs[i]['diff_c'],
                                       overlap)
        inputs[i]['nb_r'] = math.ceil(inputs[i]['diff_r'] /
                                      inputs[i]['offset_r']) + 1
        inputs[i]['nb_c'] = math.ceil(inputs[i]['diff_c'] /
                                      inputs[i]['offset_c']) + 1
    inputs = inputs[0]

    metrics = []
    outputs = []
    for i in range(len(model.outputs)):
        tshape = model.outputs[i].shape.as_list()
        name = model.outputs[i].name
        pos = min(name.index('/') if '/' in name else len(name),
                  name.index(':') if ':' in name else len(name))
        name = name[:pos]
        activation = model.get_layer(name).activation.__name__.lower()
        outputs.append({'name': name, 'shape': tshape})

        if len(tshape) == 2:
            if activation == 'softmax':
                outputs[i]['t'] = 'class'
            else:
                outputs[i]['t'] = 'multi'

            nb_classes = tshape[1]
            if nb_classes is None:
                nb_classes = model.get_layer(name).output_shape[1]
            nb_classes = int(nb_classes)
            metrics += ['%s:%s' % (name, i) for i in range(nb_classes)]

            if data_format == 'channels_first':
                shape = (nb_classes, inputs['nb_r'], inputs['nb_c'])
            elif data_format == 'channels_last':
                shape = (inputs['nb_r'], inputs['nb_c'], nb_classes)

        elif len(tshape) == 4:
            if activation == 'softmax':
                outputs[i]['t'] = 'class'
            else:
                outputs[i]['t'] = 'img'

            shape = (inputs['nb_r'], inputs['nb_c']) + tuple(tshape[1:])
        outputs[i]['p'] = np.zeros(shape)

    history, runtime = _predict_loop(model, batch_size, inputs, outputs,
                                     metrics, data_format)
    history['runtime'] = runtime

    for i in range(len(outputs)):
        if len(outputs[i]['shape']) == 2:
            if data_format == 'channels_first':
                shape = (outputs[i]['p'].shape[0], inputs['img_r'],
                         inputs['img_c'])
            elif data_format == 'channels_last':
                shape = (inputs['img_r'], inputs['img_c'],
                         outputs[i]['p'].shape[2])
        elif len(tshape) == 4:
            if data_format == 'channels_first':
                shape = (outputs[i]['p'].shape[2], inputs['img_r'],
                         inputs['img_c'])
            elif data_format == 'channels_last':
                shape = (inputs['img_r'], inputs['img_c'],
                         outputs[i]['p'].shape[4])

        count = np.zeros(shape)
        outputs[i]['img'] = np.zeros(shape)
        if len(outputs[i]['p'].shape) == 3:
            if data_format == 'channels_first':
                nb_rows = outputs[i]['p'].shape[1]
                nb_cols = outputs[i]['p'].shape[2]
            elif data_format == 'channels_last':
                nb_rows = outputs[i]['p'].shape[0]
                nb_cols = outputs[i]['p'].shape[1]
        elif len(outputs[i]['p'].shape) == 5:
            nb_rows = outputs[i]['p'].shape[0]
            nb_cols = outputs[i]['p'].shape[1]

        for j in range(nb_rows):
            for k in range(nb_cols):
                top = min(j * inputs['offset_r'],
                          inputs['img_r'] - inputs['r'])
                bottom = min(j * inputs['offset_r'] + inputs['r'],
                             inputs['img_r'])
                left = min(k * inputs['offset_c'],
                           inputs['img_c'] - inputs['c'])
                right = min(k * inputs['offset_c'] + inputs['c'],
                            inputs['img_c'])

                if data_format == 'channels_first':
                    outputs[i]['img'][:, top:bottom, left:right] += \
                        outputs[i]['p'][:, j, k]
                    count[:, top:bottom, left:right] += 1
                elif data_format == 'channels_last':
                    outputs[i]['img'][top:bottom, left:right, :] += \
                        outputs[i]['p'][j, k, :]
                    count[top:bottom, left:right, :] += 1
        outputs[i]['img'] /= count
        del outputs[i]['p']
        del outputs[i]['shape']
    return history, outputs


@ingredient.capture
@runtime
def _predict_loop(model, batch_size, inputs, outputs, metrics, data_format):
    def map_c(i, j, b, l):
        return int(((i * b) + j) / l)

    def map_r(i, j, b, l):
        return ((i * b) + j) % l

    N = inputs['nb_r'] * inputs['nb_c']
    steps = math.ceil(N / batch_size)

    history = History()
    callbacks = CallbackList([BaseLogger(), history, ProgbarLogger()])
    callbacks.set_model(model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': 1,
        'steps': steps,
        'samples': N,
        'verbose': 1,
        'do_validation': False,
        'metrics': metrics,
    })

    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)
    for b in range(steps):
        current_index = (b * batch_size) % N
        if N >= current_index + batch_size:
            current_batch_size = batch_size
        else:
            current_batch_size = N - current_index

        batch_logs = {'batch': b, 'size': current_batch_size}
        for metric in metrics:
            batch_logs[metric] = 0
        callbacks.on_batch_begin(b, batch_logs)

        bX = np.zeros((current_batch_size,) + tuple(inputs['shape'][1:]))
        for j in range(current_batch_size):
            idx_r = map_r(b, j, batch_size, inputs['nb_r'])
            idx_c = map_c(b, j, batch_size, inputs['nb_r'])
            top = min(idx_r * inputs['offset_r'],
                      inputs['img_r'] - inputs['r'])
            bottom = min(idx_r * inputs['offset_r'] + inputs['r'],
                         inputs['img_r'])
            left = min(idx_c * inputs['offset_c'],
                       inputs['img_c'] - inputs['c'])
            right = min(idx_c * inputs['offset_c'] + inputs['c'],
                        inputs['img_c'])

            if data_format == 'channels_first':
                bX[j] = inputs['img'][:, top:bottom, left:right]
            elif data_format == 'channels_last':
                bX[j] = inputs['img'][top:bottom, left:right, :]

        p = model.predict_on_batch(bX)
        if type(p) != list:
            p = [p]
        for j in range(current_batch_size):
            for i in range(len(outputs)):
                idx_r = map_r(b, j, batch_size, inputs['nb_r'])
                idx_c = map_c(b, j, batch_size, inputs['nb_r'])

                if len(outputs[i]['p'].shape) == 3:
                    if data_format == 'channels_first':
                        outputs[i]['p'][:, idx_r, idx_c] = p[i][j]
                    elif data_format == 'channels_last':
                        outputs[i]['p'][idx_r, idx_c, :] = p[i][j]
                    metric = metrics[p[i][j].argmax()]
                    batch_logs[metric] += 1. / current_batch_size
                elif len(outputs[i]['p'].shape) == 5:
                    outputs[i]['p'][idx_r, idx_c, :, :, :] = p[i][j]
        callbacks.on_batch_end(b, batch_logs)
    callbacks.on_epoch_end(0, {})
    callbacks.on_train_end()
    return history.history


@ingredient.capture
def outputs_to_img(outputs, img, base_path, rescale, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    name, ext = os.path.splitext(os.path.basename(img))
    img = load_img(img, False, rescale)
    for i in range(len(outputs)):
        if outputs[i]['t'] == 'class' or outputs[i]['t'] == 'multi':
            if data_format == 'channels_first':
                nb_classes = outputs[i]['img'].shape[0]
            elif data_format == 'channels_last':
                nb_classes = outputs[i]['img'].shape[2]

            for j in range(nb_classes):
                if data_format == 'channels_first':
                    p = outputs[i]['img'][j, :, :]
                    p = np.repeat(p.reshape((1,) + p.shape), 3, axis=0)
                elif data_format == 'channels_last':
                    p = outputs[i]['img'][:, :, j]
                    p = np.repeat(p.reshape(p.shape + (1,)), 3, axis=2)

                array_to_img(p).save(os.path.join(base_path, '%s-%s:%s%s' %
                                     (name, outputs[i]['name'], j, ext)))

                array_to_img(p * img).save(os.path.join(base_path,
                                                        '%s-%s:%s-map%s' %
                                           (name, outputs[i]['name'], j, ext)))
        elif outputs[i]['t'] == 'img':
            path = os.path.join(base_path, '%s-%s%s' %
                                (name, outputs[i]['name'], ext))
            array_to_img(outputs[i]['img']).save(path)

            path = os.path.join(base_path, '%s-%s-map%s' %
                                (name, outputs[i]['name'], ext))
            array_to_img(outputs[i]['img'] * img).save(path)
