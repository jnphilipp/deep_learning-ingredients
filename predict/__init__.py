# -*- coding: utf-8 -*-

from sacred import Ingredient

from .. import models as models_ingredient


ingredient = Ingredient('predict', ingredients=[models_ingredient.ingredient])


from .image import image
