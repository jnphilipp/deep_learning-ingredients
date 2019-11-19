# -*- coding: utf-8 -*-

from sacred import Ingredient
from tensorflow.keras.optimizers import *


ingredient = Ingredient('optimizers')


@ingredient.capture
def get(class_name: str) -> Optimizer:
    assert class_name.lower() in ['sgd', 'rmsprop', 'adagrad', 'adadelta',
                                  'adam', 'adamax', 'nadam']

    if class_name.lower() == 'sgd':
        return sgd()
    elif class_name.lower() == 'rmsprop':
        return rmsprop()
    elif class_name.lower() == 'adagrad':
        return adagrad()
    elif class_name.lower() == 'adadelta':
        return adadelta()
    elif class_name.lower() == 'adam':
        return adam()
    elif class_name.lower() == 'adamax':
        return adamax()
    elif class_name.lower() == 'nadam':
        return nadam()


@ingredient.capture
def sgd(lr: float = 0.01, momentum: float = 0.0, decay: float = 0.0,
        nesterov: float = False) -> SGD:
    return SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)


@ingredient.capture
def rmsprop(lr: float = 0.001, rho: float = 0.9, epsilon: float = None,
            decay: float = 0.0) -> RMSprop:
    return RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)


@ingredient.capture
def adagrad(lr: float = 0.01, epsilon: float = None, decay: float = 0.0) -> \
        Adagrad:
    return Adagrad(lr=lr, epsilon=epsilon, decay=decay)


@ingredient.capture
def adadelta(lr: float = 1.0, rho: float = 0.95, epsilon: float = None,
             decay: float = 0.0) -> Adadelta:
    return Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay)


@ingredient.capture
def adam(lr: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
         epsilon: float = None, decay: float = 0.0, amsgrad: bool = False) -> \
        Adam:
    return Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                decay=decay, amsgrad=amsgrad)


@ingredient.capture
def adamax(lr: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999,
           epsilon: float = None, decay: float = 0.0) -> Adamax:
    return Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                  decay=decay)


def nadam(lr: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999,
          epsilon: float = None, schedule_decay: float = 0.004) -> Nadam:
    return Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                 schedule_decay=schedule_decay)
