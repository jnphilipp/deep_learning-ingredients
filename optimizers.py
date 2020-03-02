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

from logging import Logger
from sacred import Ingredient
from tensorflow.keras.optimizers import (Adadelta, Adagrad, Adam, Adamax, Ftrl,
                                         Nadam, Optimizer, RMSprop, SGD)


ingredient = Ingredient('optimizers')


@ingredient.capture
def get(class_name: str, **kwargs) -> Optimizer:
    if class_name.lower() == 'adadelta':
        return adadelta(**kwargs)
    elif class_name.lower() == 'adagrad':
        return adagrad(**kwargs)
    elif class_name.lower() == 'adam':
        return adam(**kwargs)
    elif class_name.lower() == 'adamax':
        return adamax(**kwargs)
    elif class_name.lower() == 'ftrl':
        return ftrl(**kwargs)
    elif class_name.lower() == 'nadam':
        return nadam(**kwargs)
    elif class_name.lower() == 'rmsprop':
        return rmsprop(**kwargs)
    elif class_name.lower() == 'sgd':
        return sgd(**kwargs)


@ingredient.capture(prefix='adadelta')
def adadelta(_log: Logger, learning_rate: float = 0.001, rho: float = 0.95,
             epsilon: float = 1e-07, name: str = 'Adadelta',
             **kwargs) -> Adadelta:
    _log.info('Adadelta optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, rho={rho}, ' +
               f'epsilon={epsilon}, name={name}, kwargs={kwargs}.')
    return Adadelta(learning_rate, rho, epsilon, name, **kwargs)


@ingredient.capture(prefix='adagrad')
def adagrad(_log: Logger, learning_rate: float = 0.001,
            initial_accumulator_value: float = 0.1, epsilon: float = 1e-07,
            name: str = 'Adagrad', **kwargs) -> Adagrad:
    _log.info('Adagrad optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, ' +
               f'initial_accumulator_value={initial_accumulator_value}, ' +
               f'epsilon={epsilon}, name={name}, kwargs={kwargs}.')
    return Adagrad(learning_rate, initial_accumulator_value, epsilon, name,
                   **kwargs)


@ingredient.capture(prefix='adam')
def adam(_log: Logger, learning_rate: float = 0.001, beta_1: float = 0.9,
         beta_2: float = 0.999, epsilon: float = 1e-07, amsgrad: bool = False,
         name: str = 'Adam', **kwargs) -> Adam:
    _log.info('Adam optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, beta_1={beta_1}, ' +
               f'beta_2={beta_2}, epsilon={epsilon}, amsgrad={amsgrad}, ' +
               f'name={name}, kwargs={kwargs}.')
    return Adam(learning_rate, beta_1, beta_2, epsilon, amsgrad, name,
                **kwargs)


@ingredient.capture(prefix='adamax')
def adamax(_log: Logger, learning_rate: float = 0.001, beta_1: float = 0.9,
           beta_2: float = 0.999, epsilon: float = 1e-07, name: str = 'Adamax',
           **kwargs) -> Adamax:
    _log.info('Adamax optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, beta_1={beta_1}, ' +
               f'beta_2={beta_2}, epsilon={epsilon}, name={name}, ' +
               f'kwargs={kwargs}.')
    return Adamax(learning_rate, beta_1, beta_2, epsilon, name, **kwargs)


@ingredient.capture(prefix='ftrl')
def ftrl(_log: Logger, learning_rate: float = 0.001,
         learning_rate_power: float = -0.5,
         initial_accumulator_value: float = 0.1,
         l1_regularization_strength: float = 0.0,
         l2_regularization_strength: float = 0.0, name: str = 'Ftrl',
         l2_shrinkage_regularization_strength: float = 0.0, **kwargs) -> Ftrl:
    _log.info('Ftrl optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, ' +
               f'learning_rate_power={learning_rate_power}, ' +
               f'initial_accumulator_value={initial_accumulator_value}, ' +
               f'l1_regularization_strength={l1_regularization_strength}, ' +
               f'l2_regularization_strength={l2_regularization_strength}, ' +
               f'name={name}, l2_shrinkage_regularization_strength=' +
               f'{l2_shrinkage_regularization_strength}, kwargs={kwargs}.')
    return Ftrl(learning_rate, learning_rate_power, initial_accumulator_value,
                l1_regularization_strength, l2_regularization_strength,
                name, l2_shrinkage_regularization_strength, **kwargs)


@ingredient.capture(prefix='nadam')
def nadam(_log: Logger, learning_rate: float = 0.001, beta_1: float = 0.9,
          beta_2: float = 0.999, epsilon: float = 1e-07, name: str = 'Nadam',
          **kwargs) -> Nadam:
    _log.info('Nadam optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, beta_1={beta_1}, ' +
               f'beta_2={beta_2}, epsilon={epsilon}, name={name}, ' +
               f'kwargs={kwargs}.')
    return Nadam(learning_rate, beta_1, beta_2, epsilon, name, **kwargs)


@ingredient.capture(prefix='rmsprop')
def rmsprop(_log: Logger, learning_rate: float = 0.001, rho: float = 0.9,
            momentum: float = 0.0, epsilon: float = 1e-07,
            centered: bool = False, name: str = 'RMSprop',
            **kwargs) -> RMSprop:
    _log.info('RMSprop optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, rho={rho}, ' +
               f'momentum={momentum}, epsilon={epsilon}, ' +
               f'centered={centered}, name={name}, kwargs={kwargs}.')
    return RMSprop(learning_rate, rho, momentum, epsilon, centered, name,
                   **kwargs)


@ingredient.capture(prefix='sgd')
def sgd(_log: Logger, learning_rate: float = 0.01, momentum: float = 0.0,
        nesterov: bool = False, name: str = 'SGD', **kwargs) -> SGD:
    _log.info('SGD optimizer.')
    _log.debug(f'Config: learning_rate={learning_rate}, ' +
               f'momentum={momentum}, nesterov={nesterov}, name={name}, ' +
               f'kwargs={kwargs}.')
    return SGD(learning_rate, momentum, nesterov, name, **kwargs)
