# -*- coding: utf-8 -*-

from sacred import Ingredient
ingredient = Ingredient('predict')


from .image import image, outputs_to_img
