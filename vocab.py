# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020
#               J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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

import json

from typing import Dict, List, Optional, Union


class Vocab:
    def __init__(self, num_special: int):
        self._num_special = num_special
        self._values: Dict[str, int] = {}
        self._rvalues: Dict[int, str] = {}

    def __len__(self) -> int:
        if len(self._values) == 0:
            return 0
        else:
            return len(self._values) + self._num_special

    def add(self, k: str):
        if k not in self._values.keys():
            self._values[k] = len(self._values) + self._num_special
            self._rvalues[self.get(k)] = k

    def get(self, k: str) -> int:
        return self._values[k]

    def rget(self, k: int) -> str:
        return self._rvalues[k]

    def translate(self, text: str) -> List[int]:
        s = ''
        translated: List[int] = []
        for c in text:
            if c in self._values.keys():
                if s != '':
                    for i in range(len(s)):
                        translated.append(len(self))
                    s = ''
                translated.append(self.get(c))
            else:
                s += c
                if s in self._values.keys():
                    translated.append(self.get(c))
                    s = ''
        if s != '':
            for i in range(len(s)):
                translated.append(len(self))
        return translated

    def rtranslate(self, text: List[int], out_of_vocab_sign: str = '') -> str:
        translated = ''
        for c in text:
            if c in self._rvalues.keys():
                translated += self._rvalues[c]
            elif self._num_special < c:
                translated += out_of_vocab_sign
        return translated

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        min_idx = None
        data: Dict[str, int] = {}
        rdata: Dict[int, str] = {}
        with open(path, 'r', encoding='utf8') as f:
            for k, v in json.loads(f.read()).items():
                idx = v['id'] if isinstance(v, dict) else int(v)
                min_idx = idx if min_idx is None else min(min_idx, idx)
                data[k] = idx
                rdata[idx] = k
        v = cls(0 if min_idx is None else min_idx)
        v._values = data
        v._rvalues = rdata
        return v
