# -*- coding: utf-8 -*-
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
"""Model.metrics ingredient."""

from logging import Logger
from sacred import Ingredient
from tensorflow.keras.metrics import get as get_metric
from tensorflow.keras.metrics import (
    Accuracy,
    AUC,
    BinaryAccuracy,
    BinaryCrossentropy,
    CategoricalAccuracy,
    CategoricalCrossentropy,
    CategoricalHinge,
    CosineSimilarity,
    FalseNegatives,
    FalsePositives,
    Hinge,
    KLDivergence,
    LogCoshError,
    Mean,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanIoU,
    MeanRelativeError,
    MeanSquaredError,
    MeanSquaredLogarithmicError,
    MeanTensor,
    Metric,
    Poisson,
    Precision,
    PrecisionAtRecall,
    Recall,
    RootMeanSquaredError,
    SensitivityAtSpecificity,
    SparseCategoricalAccuracy,
    SparseCategoricalCrossentropy,
    SparseTopKCategoricalAccuracy,
    SpecificityAtSensitivity,
    SquaredHinge,
    Sum,
    TopKCategoricalAccuracy,
    TrueNegatives,
    TruePositives,
)
from typing import Any, Dict, List, Optional, Union


ingredient = Ingredient("models.metrics")


@ingredient.capture
def get(class_names: List[Union[str, Dict]], _log: Logger, **kwargs) -> List[Metric]:
    """Get metric from list."""
    metrics = []
    for i in class_names:
        class_name: str = ""
        if isinstance(i, str):
            class_name = i
        elif isinstance(i, dict):
            class_name = i["class_name"]
            for k, v in i["config"].items():
                kwargs[k] = v

        if class_name.lower() == "accuracy":
            metrics.append(accuracy(**kwargs))
        elif class_name.lower() == "auc":
            metrics.append(auc(**kwargs))
        elif class_name.lower() == "binary_accuracy":
            metrics.append(binary_accuracy(**kwargs))
        elif class_name.lower() == "binary_crossentropy":
            metrics.append(binary_crossentropy(**kwargs))
        elif class_name.lower() == "categorical_accuracy":
            metrics.append(categorical_accuracy(**kwargs))
        elif class_name.lower() == "categorical_crossentropy":
            metrics.append(categorical_crossentropy(**kwargs))
        elif class_name.lower() == "categorical_hinge":
            metrics.append(categorical_hinge(**kwargs))
        elif class_name.lower() == "cosine_similarity":
            metrics.append(cosine_similarity(**kwargs))
        elif class_name.lower() == "false_negatives":
            metrics.append(false_negatives(**kwargs))
        elif class_name.lower() == "false_positives":
            metrics.append(false_positives(**kwargs))
        elif class_name.lower() == "hinge":
            metrics.append(hinge(**kwargs))
        elif (
            class_name.lower() == "kullback_leibler_divergence"
            or class_name.lower() == "kldivergence"
            or class_name.lower() == "kld"
        ):
            metrics.append(kullback_leibler_divergence(**kwargs))
        elif class_name.lower() == "logcosh":
            metrics.append(logcosh(**kwargs))
        elif class_name.lower() == "mean":
            metrics.append(mean(**kwargs))
        elif class_name.lower() == "mean_absolute_error" or class_name.lower() == "mae":
            metrics.append(mean_absolute_error(**kwargs))
        elif (
            class_name.lower() == "mean_absolute_percentage_error"
            or class_name.lower() == "mape"
        ):
            metrics.append(mean_absolute_percentage_error(**kwargs))
        elif class_name.lower() == "meaniou":
            metrics.append(meaniou(**kwargs))
        elif class_name.lower() == "mean_relative_error":
            metrics.append(mean_relative_error(**kwargs))
        elif class_name.lower() == "mean_squared_error" or class_name.lower() == "mse":
            metrics.append(mean_squared_error(**kwargs))
        elif (
            class_name.lower() == "mean_squared_logarithmic_error"
            or class_name.lower() == "msle"
        ):
            metrics.append(mean_squared_logarithmic_error(**kwargs))
        elif class_name.lower() == "mean_tensor":
            metrics.append(mean_tensor(**kwargs))
        elif class_name.lower() == "poisson":
            metrics.append(poisson(**kwargs))
        elif class_name.lower() == "precision":
            metrics.append(precision(**kwargs))
        elif class_name.lower() == "precision_at_recall":
            metrics.append(precision_at_recall(**kwargs))
        elif class_name.lower() == "recall":
            metrics.append(recall(**kwargs))
        elif class_name.lower() == "root_mean_squared_error":
            metrics.append(root_mean_squared_error(**kwargs))
        elif class_name.lower() == "sensitivity_at_specificity":
            metrics.append(sensitivity_at_specificity(**kwargs))
        elif class_name.lower() == "sparse_categorical_accuracy":
            metrics.append(sparse_categorical_accuracy(**kwargs))
        elif class_name.lower() == "sparse_categorical_crossentropy":
            metrics.append(sparse_categorical_crossentropy(**kwargs))
        elif class_name.lower() == "sparse_top_k_categorical_accuracy":
            metrics.append(sparse_top_k_categorical_accuracy(**kwargs))
        elif class_name.lower() == "specificity_at_sensitivity":
            metrics.append(specificity_at_sensitivity(**kwargs))
        elif class_name.lower() == "sum":
            metrics.append(sum(**kwargs))
        elif class_name.lower() == "top_k_categorical_accuracy":
            metrics.append(top_k_categorical_accuracy(**kwargs))
        elif class_name.lower() == "true_negatives":
            metrics.append(true_negatives(**kwargs))
        elif class_name.lower() == "true_positives":
            metrics.append(true_positives(**kwargs))
        else:
            metrics.append(get_metric(class_name))
    return metrics


@ingredient.capture(prefix="accuracy")
def accuracy(
    _log: Logger, name: str = "accuracy", dtype: Optional[Any] = None
) -> Accuracy:
    """Accuracy."""
    _log.info("Accuracy metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return Accuracy(name, dtype)


@ingredient.capture(prefix="auc")
def auc(
    _log: Logger,
    num_thresholds: int = 200,
    curve: str = "ROC",
    summation_method: str = "interpolation",
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
    thresholds: Optional[List[float]] = None,
    multi_label: bool = False,
    label_weights: Optional[Any] = None,
) -> AUC:
    """AUC."""
    assert num_thresholds >= 1
    assert curve in ["ROC", "PR"]
    assert summation_method in ["interpolation", "minoring", "majoring"]
    _log.info("AUC metric.")
    _log.debug(
        f"Config: num_thresholds={num_thresholds}, curve={curve}, "
        + f"summation_method={summation_method}, name={name}, "
        + f"dtype={dtype}, thresholds={thresholds}, "
        + f"multi_label={multi_label}, label_weights={label_weights}."
    )
    return AUC(
        num_thresholds,
        curve,
        summation_method,
        name,
        dtype,
        thresholds,
        multi_label,
        label_weights,
    )


@ingredient.capture(prefix="binary_accuracy")
def binary_accuracy(
    _log: Logger,
    name: str = "binary_accuracy",
    dtype: Optional[Any] = None,
    threshold: float = 0.5,
) -> BinaryAccuracy:
    """BinaryAccuracy."""
    _log.info("BinaryAccuracy metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}, threshold={threshold}.")
    return BinaryAccuracy(name, dtype, threshold)


@ingredient.capture(prefix="binary_crossentropy")
def binary_crossentropy(
    _log: Logger,
    name: str = "binary_crossentropy",
    dtype: Optional[Any] = None,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
) -> BinaryCrossentropy:
    """BinaryCrossentropy."""
    _log.info("BinaryCrossentropy metric.")
    _log.debug(
        f"Config: name={name}, dtype={dtype}, "
        + f"from_logits={from_logits},"
        + f"label_smoothing={label_smoothing}."
    )
    return BinaryCrossentropy(name, dtype, from_logits, label_smoothing)


@ingredient.capture(prefix="categorical_accuracy")
def categorical_accuracy(
    _log: Logger, name: str = "categorical_accuracy", dtype: Optional[Any] = None
) -> CategoricalAccuracy:
    """CategoricalAccuracy."""
    _log.info("CategoricalAccuracy metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return CategoricalAccuracy(name, dtype)


@ingredient.capture(prefix="categorical_crossentropy")
def categorical_crossentropy(
    _log: Logger,
    name: str = "categorical_crossentropy",
    dtype: Optional[Any] = None,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
) -> CategoricalCrossentropy:
    """CategoricalCrossentropy."""
    _log.info("CategoricalCrossentropy metric.")
    _log.debug(
        f"Config: name={name}, dtype={dtype}, "
        + f"from_logits={from_logits},"
        + f"label_smoothing={label_smoothing}."
    )
    return CategoricalCrossentropy(name, dtype, from_logits, label_smoothing)


@ingredient.capture(prefix="categorical_hinge")
def categorical_hinge(
    _log: Logger, name: str = "categorical_hinge", dtype: Optional[Any] = None
) -> CategoricalHinge:
    """CategoricalHinge."""
    _log.info("CategoricalHinge metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return CategoricalHinge(name, dtype)


@ingredient.capture(prefix="cosine_similarity")
def cosine_similarity(
    _log: Logger,
    name: str = "cosine_similarity",
    dtype: Optional[Any] = None,
    axis: int = -1,
) -> CosineSimilarity:
    """CosineSimilarity."""
    _log.info("CosineSimilarity metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}, axis={axis}.")
    return CosineSimilarity(name, dtype, axis)


@ingredient.capture(prefix="false_negatives")
def false_negatives(
    _log: Logger,
    thresholds: Union[float, List[float]] = None,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> FalseNegatives:
    """FalseNegatives."""
    _log.info("FalseNegatives metric.")
    _log.debug(f"Config: thresholds={thresholds}, name={name}, dtype={dtype}.")
    return FalseNegatives(thresholds, name, dtype)


@ingredient.capture(prefix="false_positives")
def false_positives(
    _log: Logger,
    thresholds: Union[float, List[float]] = None,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> FalsePositives:
    """FalsePositives."""
    _log.info("FalsePositives metric.")
    _log.debug(f"Config: thresholds={thresholds}, name={name}, dtype={dtype}.")
    return FalsePositives(thresholds, name, dtype)


@ingredient.capture(prefix="hinge")
def hinge(_log: Logger, name: str = "hinge", dtype: Optional[Any] = None) -> Hinge:
    """Hinge."""
    _log.info("Hinge metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return Hinge(name, dtype)


@ingredient.capture(prefix="kullback_leibler_divergence")
def kullback_leibler_divergence(
    _log: Logger, name: str = "kullback_leibler_divergence", dtype: Optional[Any] = None
) -> KLDivergence:
    """KLDivergence."""
    _log.info("KLDivergence metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return KLDivergence(name, dtype)


@ingredient.capture(prefix="logcosh")
def logcosh(
    _log: Logger, name: str = "logcosh", dtype: Optional[Any] = None
) -> LogCoshError:
    """LogCoshError."""
    _log.info("LogCoshError metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return LogCoshError(name, dtype)


@ingredient.capture(prefix="mean")
def mean(_log: Logger, name: str = "mean", dtype: Optional[Any] = None) -> Mean:
    """Mean."""
    _log.info("Mean metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return Mean(name, dtype)


@ingredient.capture(prefix="mean_absolute_error")
def mean_absolute_error(
    _log: Logger, name: str = "mean_absolute_error", dtype: Optional[Any] = None
) -> MeanAbsoluteError:
    """MeanAbsoluteError."""
    _log.info("MeanAbsoluteError metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return MeanAbsoluteError(name, dtype)


@ingredient.capture(prefix="mean_absolute_percentage_error")
def mean_absolute_percentage_error(
    _log: Logger,
    name: str = "mean_absolute_percentage_" + "error",
    dtype: Optional[Any] = None,
) -> MeanAbsolutePercentageError:
    """MeanAbsolutePercentageError."""
    _log.info("MeanAbsolutePercentageError metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return MeanAbsolutePercentageError(name, dtype)


@ingredient.capture(prefix="meaniou")
def meaniou(
    _log: Logger,
    num_classes: int,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> MeanIoU:
    """MeanIoU."""
    _log.info("MeanIoU metric.")
    _log.debug(f"Config: num_classes={num_classes} name={name}, " + f"dtype={dtype}.")
    return MeanIoU(num_classes, name, dtype)


@ingredient.capture(prefix="mean_relative_error")
def mean_relative_error(
    _log: Logger,
    normalizer: Any,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> MeanRelativeError:
    """MeanRelativeError."""
    _log.info("MeanRelativeError metric.")

    _log.debug(f"Config: normalizer={normalizer}, name={name}, dtype={dtype}.")
    return MeanRelativeError(name, dtype)


@ingredient.capture(prefix="mean_squared_error")
def mean_squared_error(
    _log: Logger, name: str = "mean_squared_error", dtype: Optional[Any] = None
) -> MeanSquaredError:
    """MeanSquaredError."""
    _log.info("MeanSquaredError metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return MeanSquaredError(name, dtype)


@ingredient.capture(prefix="mean_squared_logarithmic_error")
def mean_squared_logarithmic_error(
    _log: Logger,
    name: str = "mean_squared_logarithmic_" + "error",
    dtype: Optional[Any] = None,
) -> MeanSquaredLogarithmicError:
    """MeanSquaredLogarithmicError."""
    _log.info("MeanSquaredLogarithmicError metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return MeanSquaredLogarithmicError(name, dtype)


@ingredient.capture(prefix="mean_tensor")
def mean_tensor(
    _log: Logger, name: str = "mean_tensor", dtype: Optional[Any] = None
) -> MeanTensor:
    """MeanTensor."""
    _log.info("MeanTensor metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return MeanTensor(name, dtype)


@ingredient.capture(prefix="poisson")
def poisson(
    _log: Logger, name: str = "poisson", dtype: Optional[Any] = None
) -> Poisson:
    """Poisson."""
    _log.info("Poisson metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return Poisson(name, dtype)


@ingredient.capture(prefix="precision")
def precision(
    _log: Logger,
    thresholds: Optional[Union[float, List[float]]] = None,
    top_k: Optional[int] = None,
    class_id: Optional[int] = None,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> Precision:
    """Precision."""
    _log.info("Precision metric.")
    _log.debug(
        f"Config: thresholds={thresholds}, top_k={top_k}, "
        + f"class_id={class_id}, name={name}, dtype={dtype}."
    )
    return Precision(thresholds, top_k, class_id, name, dtype)


@ingredient.capture(prefix="precision_at_recall")
def precision_at_recall(
    _log: Logger,
    recall: float,
    num_thresholds: int = 200,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> PrecisionAtRecall:
    """PrecisionAtRecall."""
    _log.info("PrecisionAtRecall metric.")
    _log.debug(
        f"Config: recall={recall}, num_thresholds={num_thresholds}, "
        + f"name={name}, dtype={dtype}."
    )
    return PrecisionAtRecall(recall, num_thresholds, name, dtype)


@ingredient.capture(prefix="recall")
def recall(
    _log: Logger,
    thresholds: Optional[Union[float, List[float]]] = None,
    top_k: Optional[int] = None,
    class_id: Optional[int] = None,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> Recall:
    """Recall."""
    _log.info("Recall metric.")
    _log.debug(
        f"Config: thresholds={thresholds}, top_k={top_k}, "
        + f"class_id={class_id}, name={name}, dtype={dtype}."
    )
    return Recall(thresholds, top_k, class_id, name, dtype)


@ingredient.capture(prefix="root_mean_squared_error")
def root_mean_squared_error(
    _log: Logger, name: str = "root_mean_squared_error", dtype: Optional[Any] = None
) -> RootMeanSquaredError:
    """RootMeanSquaredError."""
    _log.info("RootMeanSquaredError metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return RootMeanSquaredError(name, dtype)


@ingredient.capture(prefix="sensitivity_at_specificity")
def sensitivity_at_specificity(
    _log: Logger,
    specificity: float,
    num_thresholds: int = 200,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> SensitivityAtSpecificity:
    """SensitivityAtSpecificity."""
    _log.info("SensitivityAtSpecificity metric.")
    _log.debug(
        f"Config: specificity={specificity}, "
        + f"num_thresholds={num_thresholds}, name={name}, dtype={dtype}."
    )
    return SensitivityAtSpecificity(specificity, num_thresholds, name, dtype)


@ingredient.capture(prefix="sparse_categorical_accuracy")
def sparse_categorical_accuracy(
    _log: Logger, name: str = "sparse_categorical_accuracy", dtype: Optional[Any] = None
) -> SparseCategoricalAccuracy:
    """SparseCategoricalAccuracy."""
    _log.info("SparseCategoricalAccuracy metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return SparseCategoricalAccuracy(name, dtype)


@ingredient.capture(prefix="sparse_categorical_crossentropy")
def sparse_categorical_crossentropy(
    _log: Logger,
    name: str = "sparse_categorical_" + "crossentropy",
    dtype: Optional[Any] = None,
    from_logits: bool = False,
    axis: int = -1,
) -> SparseCategoricalCrossentropy:
    """SparseCategoricalCrossentropy."""
    _log.info("SparseCategoricalCrossentropy metric.")
    _log.debug(
        f"Config: name={name}, dtype={dtype}, "
        + f"from_logits={from_logits}, axis={axis}."
    )
    return SparseCategoricalCrossentropy(name, dtype, from_logits, axis)


@ingredient.capture(prefix="sparse_top_k_categorical_accuracy")
def sparse_top_k_categorical_accuracy(
    _log: Logger,
    k: int = 5,
    name: str = "sparse_top_k_categorical_" + "accuracy",
    dtype: Optional[Any] = None,
) -> SparseTopKCategoricalAccuracy:
    """SparseTopKCategoricalAccuracy."""
    _log.info("SparseTopKCategoricalAccuracy metric.")
    _log.debug(f"Config: k={k}, name={name}, dtype={dtype}.")
    return SparseTopKCategoricalAccuracy(k, name, dtype)


@ingredient.capture(prefix="specificity_at_sensitivity")
def specificity_at_sensitivity(
    _log: Logger,
    sensitivity: float,
    num_thresholds: int = 200,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> SpecificityAtSensitivity:
    """SpecificityAtSensitivity."""
    _log.info("SpecificityAtSensitivity metric.")
    _log.debug(
        f"Config: sensitivity={sensitivity}, "
        + f"num_thresholds={num_thresholds}, name={name}, dtype={dtype}."
    )
    return SpecificityAtSensitivity(sensitivity, num_thresholds, name, dtype)


@ingredient.capture(prefix="squared_hinge")
def squared_hinge(
    _log: Logger, name: str = "squared_hinge", dtype: Optional[Any] = None
) -> SquaredHinge:
    """SquaredHinge."""
    _log.info("SquaredHinge metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return SquaredHinge(name, dtype)


@ingredient.capture(prefix="sum")
def sum(_log: Logger, name: str = "sum", dtype: Optional[Any] = None) -> Sum:
    """Sum."""
    _log.info("Sum metric.")
    _log.debug(f"Config: name={name}, dtype={dtype}.")
    return Sum(name, dtype)


@ingredient.capture(prefix="top_k_categorical_accuracy")
def top_k_categorical_accuracy(
    _log: Logger,
    k: int = 5,
    name: str = "top_k_categorical_accuracy",
    dtype: Optional[Any] = None,
) -> TopKCategoricalAccuracy:
    """TopKCategoricalAccuracy."""
    _log.info("TopKCategoricalAccuracy metric.")
    _log.debug(f"Config: k={k}, name={name}, dtype={dtype}.")
    return TopKCategoricalAccuracy(k, name, dtype)


@ingredient.capture(prefix="true_negatives")
def true_negatives(
    _log: Logger,
    thresholds: Optional[Union[float, List[float]]] = None,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> TrueNegatives:
    """TrueNegatives."""
    _log.info("TrueNegatives metric.")
    _log.debug(f"Config: thresholds={thresholds}, name={name}, dtype={dtype}.")
    return TrueNegatives(thresholds, name, dtype)


@ingredient.capture(prefix="true_positives")
def true_positives(
    _log: Logger,
    thresholds: Optional[Union[float, List[float]]] = None,
    name: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> TruePositives:
    """TruePositives."""
    _log.info("TruePositives metric.")
    _log.debug(f"Config: thresholds={thresholds}, name={name}, dtype={dtype}.")
    return TruePositives(thresholds, name, dtype)
