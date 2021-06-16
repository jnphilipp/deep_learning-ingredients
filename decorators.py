# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2019-2021 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
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
"""Decorators."""

import time

from functools import wraps
from typing import Callable, Tuple


def runtime(func: Callable) -> Callable:
    """Runtime decorator."""

    @wraps(func)
    def func_wrapper(*args, **kwargs) -> Tuple:
        start_time = int(round(time.time() * 1000))
        values = func(*args, **kwargs)
        runtime = (int(round(time.time() * 1000)) - start_time) / 1000.0
        if type(values) == tuple:
            values += (runtime,)
        else:
            values = (values, runtime)
        return values

    return func_wrapper
