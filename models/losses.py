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
"""Models.losses ingredient."""

from logging import Logger
from sacred import Ingredient
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
    CategoricalHinge,
    CosineSimilarity,
    Hinge,
    Huber,
    KLDivergence,
    LogCosh,
    Loss,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogarithmicError,
    Poisson,
    Reduction,
    SparseCategoricalCrossentropy,
    SquaredHinge,
)


ingredient = Ingredient("models.losses")


@ingredient.capture
def get(class_name: str, _log: Logger, **kwargs) -> Loss:
    """Get loss function."""
    if class_name.lower() == "binary_crossentropy":
        return binary_crossentropy(**kwargs)
    elif class_name.lower() == "categorical_crossentropy":
        return categorical_crossentropy(**kwargs)
    elif class_name.lower() == "categorical_hinge":
        return categorical_hinge(**kwargs)
    elif class_name.lower() == "cosine_similarity":
        return cosine_similarity(**kwargs)
    elif class_name.lower() == "hinge":
        return hinge(**kwargs)
    elif class_name.lower() == "huber":
        return huber(**kwargs)
    elif (
        class_name.lower() == "kullback_leibler_divergence"
        or class_name.lower() == "kldivergence"
        or class_name.lower() == "kld"
    ):
        return kullback_leibler_divergence(**kwargs)
    elif class_name.lower() == "logcosh":
        return logcosh(**kwargs)
    elif class_name.lower() == "mean_absolute_error" or class_name.lower() == "mae":
        return mean_absolute_error(**kwargs)
    elif (
        class_name.lower() == "mean_absolute_percentage_error"
        or class_name.lower() == "mape"
    ):
        return mean_absolute_percentage_error(**kwargs)
    elif class_name.lower() == "mean_squared_error" or class_name.lower() == "mse":
        return mean_squared_error(**kwargs)
    elif (
        class_name.lower() == "mean_squared_logarithmic_error"
        or class_name.lower() == "msle"
    ):
        return mean_squared_logarithmic_error(**kwargs)
    elif class_name.lower() == "poisson":
        return poisson(**kwargs)
    elif class_name.lower() == "sparse_categorical_crossentropy":
        return sparse_categorical_crossentropy(**kwargs)
    elif class_name.lower() == "squared_hinge":
        return squared_hinge(**kwargs)
    else:
        return get_loss(class_name)


@ingredient.capture(prefix="binary_crossentropy")
def binary_crossentropy(
    _log: Logger,
    from_logits: bool = False,
    label_smoothing: float = 0,
    reduction: str = Reduction.AUTO,
    name: str = "binary_crossentropy",
) -> BinaryCrossentropy:
    """Binary crossentropy loss."""
    _log.info("BinaryCrossentropy loss.")
    _log.debug(
        f"Config: from_logits={from_logits}, "
        + f"label_smoothing={label_smoothing}, reduction={reduction}, "
        + f"name={name}."
    )
    return BinaryCrossentropy(from_logits, label_smoothing, reduction, name)


@ingredient.capture(prefix="binary_focal_crossentropy")
def binary_focal_crossentropy(
    _log: Logger,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction: str = Reduction.AUTO,
    name: str = "binary_focal_crossentropy",
) -> BinaryFocalCrossentropy:
    """Binary focal crossentropy loss."""
    _log.info("BinaryFocalCrossentropy loss.")
    _log.debug(
        f"Config: gamma={gamma}, from_logits={from_logits}, "
        + f"label_smoothing={label_smoothing}, axis={axis}, reduction={reduction}, "
        + f"name={name}."
    )
    return BinaryFocalCrossentropy(
        gamma,
        from_logits,
        label_smoothing,
        axis,
        reduction,
        name,
    )


@ingredient.capture(prefix="categorical_crossentropy")
def categorical_crossentropy(
    _log: Logger,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
    reduction: str = Reduction.AUTO,
    name="categorical_crossentropy",
) -> CategoricalCrossentropy:
    """Categorical crossentropy loss."""
    _log.info("CategoricalCrossentropy loss.")
    _log.debug(
        f"Config: from_logits={from_logits}, label_smoothing={label_smoothing}, "
        + f"axis={axis}, reduction={reduction}, name={name}."
    )
    return CategoricalCrossentropy(from_logits, label_smoothing, axis, reduction, name)


@ingredient.capture(prefix="categorical_hinge")
def categorical_hinge(
    _log: Logger, reduction: str = Reduction.AUTO, name: str = "categorical_hinge"
) -> CategoricalHinge:
    """Categorical hinge loss."""
    _log.info("CategoricalHinge loss.")
    _log.debug(f"Config: reduction={reduction} name={name}.")
    return CategoricalHinge(reduction, name)


@ingredient.capture(prefix="cosine_similarity")
def cosine_similarity(
    _log: Logger,
    axis: int = -1,
    reduction: str = Reduction.AUTO,
    name: str = "cosine_similarity",
) -> CosineSimilarity:
    """Cosine similarity loss."""
    _log.info("CosineSimilarity loss.")
    _log.debug(f"Config: axis={axis}, reduction={reduction}, name={name}.")
    return CosineSimilarity(axis, reduction, name)


@ingredient.capture(prefix="hinge")
def hinge(_log: Logger, reduction: str = Reduction.AUTO, name: str = "hinge") -> Hinge:
    """Hinge loss."""
    _log.info("Hinge loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return Hinge(reduction, name)


@ingredient.capture(prefix="huber")
def huber(
    _log: Logger,
    delta: float = 1.0,
    reduction: str = Reduction.AUTO,
    name: str = "huber_loss",
) -> Huber:
    """Huber loss."""
    _log.info("Huber loss.")
    _log.debug(f"Config: delta={delta}, reduction={reduction}, name={name}.")
    return Huber(delta, reduction, name)


@ingredient.capture(prefix="kullback_leibler_divergence")
def kullback_leibler_divergence(
    _log: Logger, reduction: str = Reduction.AUTO, name="kullback_leibler_divergence"
) -> KLDivergence:
    """Kullback-Leibler divergence loss."""
    _log.info("KLDivergence loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return KLDivergence(reduction, name)


@ingredient.capture(prefix="logcosh")
def logcosh(
    _log: Logger, reduction: str = Reduction.AUTO, name: str = "logcosh"
) -> LogCosh:
    """Log cosh loss."""
    _log.info("LogCosh loss.")
    _log.debug(f"Config: reduction={reduction} name={name}.")
    return LogCosh(reduction, name)


@ingredient.capture(prefix="mean_absolute_error")
def mean_absolute_error(
    _log: Logger, reduction: str = Reduction.AUTO, name: str = "mean_absolute_error"
) -> MeanAbsoluteError:
    """Mean absolute error loss."""
    _log.info("MeanAbsoluteError loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return MeanAbsoluteError(reduction, name)


@ingredient.capture(prefix="mean_absolute_percentage_error")
def mean_absolute_percentage_error(
    _log: Logger, reduction: str = Reduction.AUTO, name="mean_absolute_percentage_error"
) -> MeanAbsolutePercentageError:
    """Mean absolute percentage error loss."""
    _log.info("MeanAbsolutePercentageError loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return MeanAbsolutePercentageError(reduction, name)


@ingredient.capture(prefix="mean_squared_error")
def mean_squared_error(
    _log: Logger, reduction: str = Reduction.AUTO, name="mean_squared_error"
) -> MeanSquaredError:
    """Mean squared error loss."""
    _log.info("MeanSquaredError loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return MeanSquaredError(reduction, name)


@ingredient.capture(prefix="mean_squared_logarithmic_error")
def mean_squared_logarithmic_error(
    _log: Logger, reduction: str = Reduction.AUTO, name="mean_squared_logarithmic_error"
) -> MeanSquaredLogarithmicError:
    """Mean squared logarithmic error loss."""
    _log.info("MeanSquaredLogarithmicError loss.")
    _log.debug(f"Config: reduction={reduction} name={name}.")
    return MeanSquaredLogarithmicError(reduction, name)


@ingredient.capture(prefix="poisson")
def poisson(
    _log: Logger, reduction: str = Reduction.AUTO, name: str = "poisson"
) -> Poisson:
    """Poisson loss."""
    _log.info("Poisson loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return Poisson(reduction, name)


@ingredient.capture(prefix="sparse_categorical_crossentropy")
def sparse_categorical_crossentropy(
    _log: Logger,
    from_logits: bool = False,
    reduction: str = Reduction.AUTO,
    name: str = "sparse_categorical_" + "crossentropy",
) -> SparseCategoricalCrossentropy:
    """Sparse categorical crossentropy loss."""
    _log.info("SparseCategoricalCrossentropy loss.")
    _log.debug(
        f"Config: from_logits={from_logits}, reduction={reduction}, " + f"name={name}."
    )
    return SparseCategoricalCrossentropy(from_logits, reduction, name)


@ingredient.capture(prefix="squared_hinge")
def squared_hinge(
    _log: Logger, reduction: str = Reduction.AUTO, name: str = "squared_hinge"
) -> SquaredHinge:
    """Squared hinge loss."""
    _log.info("SquaredHinge loss.")
    _log.debug(f"Config: reduction={reduction}, name={name}.")
    return SquaredHinge(reduction, name)
