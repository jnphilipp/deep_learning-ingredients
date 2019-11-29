# -*- coding: utf-8 -*-
# Copyright (C) 2019 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of deep_learning-ingredients.
#
# deep_learning-ingredients is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# deep_learning-ingredients is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with deep_learning-ingredients. If not, see
# <http://www.gnu.org/licenses/>.

from csv import DictWriter
from tensorflow.keras.callbacks import Callback


class WeightsLogging(Callback):
    def __init__(self, mode='epochs', path=None):
        assert mode in ['batches', 'epochs']
        super(WeightsLogging, self).__init__()
        self.mode = mode
        self.path = path
        self.weights_history = {
            'mode': self.mode,
            'layers': []
        }

    def _log_weights(self, i):
        row = {self.mode: i}
        for layer in self.model.layers:
            row[layer.name] = '|'.join(','.join([f'min:{float(w.min())}',
                                                 f'max:{float(w.max())}',
                                                 f'mean:{float(w.mean())}',
                                                 f'std:{float(w.std())}'])
                                       for w in layer.get_weights())
        with open(self.path, 'a', encoding='utf8') as f:
            writer = DictWriter(f, self.fields, dialect='unix')
            writer.writerow(row)

    def on_train_begin(self, logs=None):
        self.fields = [self.mode] + [layer.name for layer in self.model.layers]
        with open(self.path, 'w', encoding='utf8') as f:
            writer = DictWriter(f, self.fields, dialect='unix')
            writer.writeheader()

    def on_train_batch_end(self, batch, logs=None):
        if self.mode == 'batches':
            self._log_weights(batch)

    def on_epoch_end(self, epoch, logs=None):
        if self.mode == 'epochs':
            self._log_weights(epoch)
