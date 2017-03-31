# -*- coding: utf-8 -*-

import os
import sys

from json import dumps
from keras.models import load_model
from keras.utils import plot_model
from sacred import Ingredient


ingredients = Ingredient('model')


@ingredients.config
def config():
    summary = False


@ingredients.capture
def get(build_func, summary, path=None, *args, **kwargs):
    if build_func and not path:
        model = build_func(*args, **kwargs)
    else:
        model = load()
    if summary:
        model.summary()
    return model


@ingredients.capture
def load(path):
    print('Loading model [%s]...' % path)
    return load_model(path)


@ingredients.capture
def save(path, model, name=None):
    with open(os.path.join(path, '%s.json' % (name if name else 'model')), 'w', encoding='utf8') as f:
        f.write(model.to_json())
        f.write('\n')

    stdout = sys.stdout
    with open(os.path.join(path, '%ssummary' % ('%s_' % name if name else '')), 'w', encoding='utf8') as f:
        sys.stdout = f
        model.summary()
    sys.stdout = stdout

    model.save(os.path.join(path, '%s.h5' % (name if name else 'model')))
