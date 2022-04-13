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
"""Vocab."""

import json

from typing import Dict, List


class Vocab:
    """Vocab."""

    def __init__(self, num_special: int):
        """Init."""
        self._num_special = num_special
        self._values: Dict[str, int] = {}
        self._rvalues: Dict[int, str] = {}

    def __len__(self) -> int:
        """Length."""
        if len(self._values) == 0:
            return 0
        else:
            return len(self._values) + self._num_special

    def add(self, k: str):
        """Add."""
        if k not in self._values.keys():
            self._values[k] = len(self._values) + self._num_special
            self._rvalues[self.get(k)] = k

    def get(self, k: str) -> int:
        """Get."""
        return self._values[k]

    def rget(self, k: int, out_of_vocab_sign: str = "") -> str:
        """Reverse get."""
        if k in self._rvalues:
            return self._rvalues[k]
        else:
            return out_of_vocab_sign

    def translate(self, text: str) -> List[int]:
        """Translate."""
        s = ""
        translated: List[int] = []
        for c in text:
            if c in self._values.keys():
                if s != "":
                    for i in range(len(s)):
                        translated.append(len(self))
                    s = ""
                translated.append(self.get(c))
            else:
                s += c
                if s in self._values.keys():
                    translated.append(self.get(c))
                    s = ""
        if s != "":
            for i in range(len(s)):
                translated.append(len(self))
        return translated

    def rtranslate(self, text: List[int], out_of_vocab_sign: str = "") -> str:
        """Reverse translate."""
        translated = ""
        for c in text:
            if c in self._rvalues.keys():
                translated += self._rvalues[c]
            elif self._num_special < c:
                translated += out_of_vocab_sign
        return translated

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Load from file."""
        min_idx = None
        data: Dict[str, int] = {}
        rdata: Dict[int, str] = {}
        with open(path, "r", encoding="utf8") as f:
            for k, v in json.loads(f.read()).items():
                idx = v["id"] if isinstance(v, dict) else int(v)
                if min_idx is None:
                    min_idx = idx
                else:
                    min_idx = min(min_idx, idx)
                data[k] = idx
                rdata[idx] = k
        v = cls(0 if min_idx is None else min_idx)
        v._values = data
        v._rvalues = rdata
        return v
