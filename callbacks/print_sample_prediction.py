# -*- coding: utf-8 -*-

from tensorflow.keras.callbacks import Callback


class PrintSamplePrediction(Callback):
    def __init__(self, vocab, X, y):
        super(PrintSamplePrediction, self).__init__()
        self.vocab = vocab
        self.X = X
        self.y = y

    def trans_output(self, v):
        if len(v.shape) == 1:
            return ''.join([self.vocab[int(v.argmax())]])
        else:
            return ''.join([self.vocab[int(l)] for l in v.argmax(axis=1)])

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X)

        print('\r', end='')
        for i in range(len(self.X)):
            if type(self.y) == list:
                outs = []
                ys = []
                for j in range(len(self.y)):
                    outs.append(self.trans_output(p[j][i]))
                    ys.append(self.trans_output(y[j][i]))
            else:
                outs = [self.trans_output(p[i])]
                ys = [self.trans_output(self.y[i])]

            s = f'"{"".join([self.vocab[int(l)] for l in self.X[i]])}" =>'
            for o, y in zip(outs, ys):
                s += f' "{o}":"{y}"'
            print(s)
