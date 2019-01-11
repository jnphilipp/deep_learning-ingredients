# -*- coding: utf-8 -*-

import math

from generators import CTCImageDataGenerator, PatchImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

from .core import from_directory
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
        validation_datagen = ImageDataGenerator(
            **validation_image_datagen_args)
        validation_generator = validation_datagen.flow_from_directory(
            validation_set, size, batch_size=batch_size, class_mode=class_mode)
        validation_steps = math.ceil(validation_generator.samples / batch_size)
    else:
        validation_generator = None
        validation_steps = None

    return train_generator, train_steps, validation_generator, validation_steps


@ingredient.capture
def patchdatagen_from_directory(shape, train_set, train_samples, batch_size,
                                train_patch_image_datagen_args, validation_set,
                                validation_samples,
                                validation_patch_image_datagen_args={}):
    X, y = from_directory(which_set=train_set)

    train_datagen = PatchImageDataGenerator(**train_patch_image_datagen_args)
    train_generator = train_datagen.flow(X, y, shape, train_samples,
                                         batch_size)
    train_steps = math.ceil(train_samples / batch_size)

    if validation_set and validation_samples and \
            validation_patch_image_datagen_args:
        X, y = from_directory(which_set=validation_set)
        validation_datagen = PatchImageDataGenerator(
            **validation_patch_image_datagen_args)
        validation_generator = validation_datagen.flow(X, y, shape,
                                                       validation_samples,
                                                       batch_size)
        validation_steps = math.ceil(validation_samples / batch_size)
    else:
        validation_generator = None
        validation_steps = None

    return train_generator, train_steps, validation_generator, validation_steps


@ingredient.capture
def ctc_datagen(shape, train_samples, batch_size, empty_images, create_masks,
                train_ctc_image_datagen_args, validation_samples,
                background_dataset, background_which_set,
                validation_ctc_image_datagen_args={},
                class_mode='categorical'):
    background_images = from_directory(dataset=background_dataset,
                                       which_set=background_which_set)
    background_images = [img['img'] for img in background_images]

    train_datagen = CTCImageDataGenerator(**train_ctc_image_datagen_args)
    train_generator = train_datagen.flow(shape, train_samples, batch_size,
                                         empty_images, create_masks,
                                         background_images, class_mode)
    train_steps = math.ceil(train_samples / batch_size)

    if validation_ctc_image_datagen_args and validation_samples:
        validation_datagen = CTCImageDataGenerator(
            **validation_ctc_image_datagen_args)
        validation_generator = validation_datagen.flow(shape, train_samples,
                                                       batch_size,
                                                       empty_images,
                                                       create_masks,
                                                       background_images,
                                                       class_mode)
        validation_steps = math.ceil(validation_samples / batch_size)

    return train_generator, train_steps, validation_generator, validation_steps
