# -*- coding: utf-8 -*-

import math

from keras.preprocessing.image import ImageDataGenerator

from .. import ingredient, get_full_dataset_path


@ingredient.capture
def datagen_from_directory(size, batch_size, train_image_datagen_args,
                           validation_image_datagen_args={},
                           class_mode='categorical'):
    train_set, validation_set = get_full_dataset_path()

    train_datagen = ImageDataGenerator(**train_image_datagen_args)
    train_generator = train_datagen.flow_from_directory(train_set, size,
                                                        batch_size=batch_size,
                                                        class_mode=class_mode)
    train_steps = math.ceil(train_generator.samples / batch_size)

    if validation_set:
        validation_datagen = ImageDataGenerator(**validation_image_datagen_args)
        validation_generator = validation_datagen.flow_from_directory(
            validation_set, size, batch_size=batch_size, class_mode=class_mode)
        validation_steps = math.ceil(validation_generator.samples / batch_size)
    else:
        validation_generator = None
        validation_steps = None

    return train_generator, train_steps, validation_generator, validation_steps