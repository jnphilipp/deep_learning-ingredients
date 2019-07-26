# -*- coding: utf-8 -*-

from keras.optimizers import *
from sacred import Ingredient

ingredient = Ingredient('optimizers')


@ingredient.capture
def get(class_name):
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
def sgd(lr=0.01, momentum=0.0, decay=0.0, nesterov=False):
    return SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)


@ingredient.capture
def rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0):
    return RMSProp(lr=lr, rho=rho, epsilon=epsilon, decay=decay)


@ingredient.capture
def adagrad(lr=0.01, epsilon=None, decay=0.0):
    return Adagrad(lr=lr, epsilon=epsilon, decay=decay)


@ingredient.capture
def adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0):
    return Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay)


@ingredient.capture
def adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
         amsgrad=False):
    return Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                decay=decay, amsgrad=amsgrad)


@ingredient.capture
def adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0):
    return adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                  decay=decay)


def nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None,
          schedule_decay=0.004):
    return Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                 schedule_decay=schedule_decay)
