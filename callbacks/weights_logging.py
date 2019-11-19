# -*- coding: utf-8 -*-

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
