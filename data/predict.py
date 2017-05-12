# -*- coding: utf-8 -*-

import math
import numpy as np

from ingredients.data import ingredients
from keras import backend as K
from keras.callbacks import BaseLogger, CallbackList, History, ProgbarLogger

from utils import current_time_millis
from utils.images import offset


@ingredients.capture
def image(model, image, batch_size, overlap, metrics):
    (input_r, input_c, image_r, image_c, diff_r, diff_c,
        offset_r, offset_c) = offset(model, image, overlap)
    t = np.zeros((model.get_layer('p').output_shape[1],
                  math.ceil(diff_r / offset_r) + 1,
                  math.ceil(diff_c / offset_c) + 1))

    mask_layer = model.get_layer('mask')
    if mask_layer:
        tm = np.zeros((math.ceil(diff_r / offset_r) + 1,
                       math.ceil(diff_c / offset_c) + 1) + mask_layer.output_shape[1:])
    samples = t.shape[1] * t.shape[2]

    def map_r(i, j, b, l):
        return ((i * b) + j) % l

    def map_c(i, j, b, l):
        return int(((i * b) + j) / l)

    history = History()
    callbacks = CallbackList([BaseLogger(), history, ProgbarLogger()])
    callbacks.set_model(model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': 1,
        'samples': samples,
        'verbose': 1,
        'do_validation': False,
        'metrics': metrics,
    })

    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)
    start_time = current_time_millis()
    for i in range(math.ceil(samples / batch_size)):
        current_index = (i * batch_size) % samples
        current_batch_size = batch_size if samples >= current_index + batch_size \
            else samples - current_index

        batch_logs = {'batch': i, 'size': current_batch_size}
        for metric in metrics:
            batch_logs[metric] = 0
        callbacks.on_batch_begin(i, batch_logs)

        bX = np.zeros((current_batch_size,) + model.get_layer('input').input_shape[1:4])
        for j in range(current_batch_size):
            idx_r = map_r(i, j, batch_size, t.shape[1])
            idx_c = map_c(i, j, batch_size, t.shape[1])
            top_r = min(idx_r * offset_r, image_r - input_r)
            bottom_r = min(idx_r * offset_r + input_r, image_r)
            left_c = min(idx_c * offset_c, image_c - input_c)
            right_c = min(idx_c * offset_c + input_c, image_c)

            if K.image_data_format() == 'channels_first':
                bX[j] = image[:, top_r:bottom_r, left_c:right_c]
            else:
                bX[j] = image[top_r:bottom_r, left_c:right_c, :]

        p = model.predict(bX, batch_size=batch_size)
        for j in range(current_batch_size):
            if mask_layer:
                t[:,
                  map_r(i, j, batch_size, t.shape[1]),
                  map_c(i, j, batch_size, t.shape[1])] = p[0][j]
                tm[map_r(i, j, batch_size, t.shape[1]),
                   map_c(i, j, batch_size, t.shape[1]), :, :, :] = p[1][j]
                batch_logs[metrics[p[0][j].argmax()]] += 1. / current_batch_size
            else:
                t[:,
                  map_r(i, j, batch_size, t.shape[1]),
                  map_c(i, j, batch_size, t.shape[1])] = p[j]
                batch_logs[metrics[p[j].argmax()]] += 1. / current_batch_size
        callbacks.on_batch_end(i, batch_logs)
    callbacks.on_epoch_end(0, {'runtime': (current_time_millis() - start_time) / 1000})
    callbacks.on_train_end()

    if mask_layer:
        return t, tm
    else:
        return t
