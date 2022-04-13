# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2019-2022 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of deep_learning-ingredients.
#
# deep_learning-ingredients is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# deep_learning-ingredients is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with deep_learning-ingredients. If not, see <http://www.gnu.org/licenses/>.
"""Print sample prediction callback."""

from tensorflow.keras.callbacks import Callback


class PrintSamplePrediction(Callback):
    """Print sample prediction callback."""

    def __init__(self, x, y, translate_func):
        """Init."""
        super().__init__()
        self.x = x
        self.y = y
        self.translate_func = translate_func

    def _trans_output(self, v):
        if len(v.shape) == 1:
            return self.translate_func([int(v.argmax())])
        else:
            return self.translate_func([i for i in v.argmax(axis=1)])

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end."""
        p = self.model.predict(self.x)

        print("\r", end="")
        for i in range(len(self.x)):
            if type(self.y) == list:
                outs = []
                ys = []
                for j in range(len(self.y)):
                    outs.append(self._trans_output(p[j][i]))
                    ys.append(self._trans_output(self.y[j][i]))
            else:
                outs = [self._trans_output(p[i])]
                ys = [self._trans_output(self.y[i])]

            s = f'"{self.translate_func(self.x[i])}" =>'
            for o, y in zip(outs, ys):
                s += f' "{o}":"{y}"'
            print(s)
