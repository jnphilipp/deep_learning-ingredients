# -*- coding: utf-8 -*-

import math
import numpy as np
import os

from decorators import runtime
from ingredients import models
from ingredients.datasets.h5py import save
from ingredients.datasets.images import load
from keras import backend as K
from keras.callbacks import BaseLogger, CallbackList, History, ProgbarLogger
from keras.preprocessing.image import array_to_img

from . import ingredient


@ingredient.command
def images(images, _log=None, _run=None):
    """Visualize predictions from images."""
    if isinstance(images, str):
        images = [images]

    base_dir = os.path.join(_run.observers[0].run_dir, 'images')
    os.makedirs(base_dir, exist_ok=True)

    model = models.load()
    for img in images:
        name, ext = os.path.splitext(os.path.basename(img))
        _log.info(f'Image: {name}')

        history, outputs = func(model, img)

        matrices = {o['name']: o['img'] for o in outputs}
        save(os.path.join(base_dir, 'probabilities.h5'), name, matrices)
        outputs_to_img(outputs, img, base_dir)


@ingredient.capture
def func(model, image_path, batch_size, overlap, rescale, data_format=None):
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
            if inputs[i]['shape'][1] == 1:
                inputs[i]['color_mode'] = 'grayscale'
            elif inputs[i]['shape'][1] == 3:
                inputs[i]['color_mode'] = 'rgb'
            elif inputs[i]['shape'][1] == 4:
                inputs[i]['color_mode'] = 'rgba'
            inputs[i]['r'] = inputs[i]['shape'][2]
            inputs[i]['c'] = inputs[i]['shape'][3]
        elif data_format == 'channels_last':
            inputs[i]['r'] = inputs[i]['shape'][1]
            inputs[i]['c'] = inputs[i]['shape'][2]
            if inputs[i]['shape'][3] == 1:
                inputs[i]['color_mode'] = 'grayscale'
            elif inputs[i]['shape'][3] == 3:
                inputs[i]['color_mode'] = 'rgb'
            elif inputs[i]['shape'][3] == 4:
                inputs[i]['color_mode'] = 'rgba'

        inputs[i]['img'] = load(image_path, inputs[i]['color_mode']) * rescale
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
                    cur_img_shape = outputs[i]['img'][:, top:bottom,
                                                      left:right].shape
                    cur_p_shape = outputs[i]['p'][:, j, k].shape
                    if len(cur_p_shape) != 1 and cur_img_shape != cur_p_shape:
                        repeat1 = math.floor(cur_img_shape[1] / cur_p_shape[1])
                        repeat2 = math.floor(cur_img_shape[2] / cur_p_shape[2])
                        outputs[i]['img'][:, top:bottom, left:right] += \
                            np.repeat(np.repeat(outputs[i]['p'][:, j, k],
                                                repeat1, 1), repeat2, 2)
                    else:
                        outputs[i]['img'][:, top:bottom, left:right] += \
                            outputs[i]['p'][:, j, k]
                    count[:, top:bottom, left:right] += 1
                elif data_format == 'channels_last':
                    cur_img_shape = outputs[i]['img'][top:bottom, left:right,
                                                      :].shape
                    cur_p_shape = outputs[i]['p'][j, k, :].shape
                    if len(cur_p_shape) != 1 and cur_img_shape != cur_p_shape:
                        repeat0 = math.floor(cur_img_shape[0] / cur_p_shape[0])
                        repeat1 = math.floor(cur_img_shape[1] / cur_p_shape[1])
                        outputs[i]['img'][top:bottom, left:right, :] += \
                            np.repeat(np.repeat(outputs[i]['p'][j, k, :],
                                                repeat0, 0), repeat1, 1)
                    else:
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
    img = load(img) * rescale
    for i in range(len(outputs)):
        output_name = outputs[i]['name']

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

                path = os.path.join(base_path,
                                    f'{name}-{output_name}:{j}{ext}')
                array_to_img(p * 255, scale=False).save(path)

                path = os.path.join(base_path,
                                    f'{name}-{output_name}:{j}-image{ext}')
                array_to_img(p * img * 255, scale=False).save(path)
        elif outputs[i]['t'] == 'img':
            path = os.path.join(base_path, f'{name}-{output_name}{ext}')
            array_to_img(outputs[i]['img']).save(path)

            path = os.path.join(base_path, f'{name}-{output_name}-image{ext}')
            array_to_img(outputs[i]['img'] * img).save(path)
