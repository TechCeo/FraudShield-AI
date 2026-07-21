"""Probability, threshold, and fraud-detection evaluation contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    """A decision threshold selected from a validation probability stream."""

    threshold: float
    beta: float
    fbeta: float
    recall: float
    precision: float
    predicted_positive: int


def _binary_target(values: Sequence[int] | np.ndarray) -> np.ndarray:
    target = np.asarray(values)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("target must be a nonempty one-dimensional array")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("target must contain only 0 and 1")
    target = target.astype(np.int8, copy=False)
    if np.unique(target).size != 2:
        raise ValueError("target must contain both classes")
    return target


def _probabilities(values: Sequence[float] | np.ndarray, rows: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (rows,):
        raise ValueError("probabilities must align one-to-one with target rows")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return probabilities


def select_fbeta_threshold(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    beta: float = 2.0,
) -> ThresholdSelection:
    """Select the validation threshold maximizing F-beta.

    Ties favor higher recall, then higher precision, then the higher threshold.
    The threshold is intended to be frozen before holdout evaluation.
    """

    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")
    target_array = _binary_target(target)
    probability_array = _probabilities(probabilities, len(target_array))
    precision, recall, thresholds = precision_recall_curve(
        target_array, probability_array
    )
    if thresholds.size == 0:
        raise ValueError("at least two distinct probability outcomes are required")
    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    scores = np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    order = np.lexsort((thresholds, precision, recall, scores))
    position = int(order[-1])
    threshold = float(thresholds[position])
    return ThresholdSelection(
        threshold=threshold,
        beta=float(beta),
        fbeta=float(scores[position]),
        recall=float(recall[position]),
        precision=float(precision[position]),
        predicted_positive=int(np.count_nonzero(probability_array >= threshold)),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _operating_metrics(
    target: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    prediction = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    precision = _safe_ratio(int(tp), int(tp + fp))
    recall = _safe_ratio(int(tp), int(tp + fn))
    specificity = _safe_ratio(int(tn), int(tn + fp))
    f1 = _safe_ratio(2 * int(tp), 2 * int(tp) + int(fp) + int(fn))
    f2 = _safe_ratio(5 * int(tp), 5 * int(tp) + 4 * int(fn) + int(fp))
    return {
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "false_negative_rate": 1.0 - recall,
        "f1": f1,
        "f2": f2,
        "alert_rate": float(prediction.mean()),
    }


def _precision_at_recall(
    precision: np.ndarray, recall: np.ndarray, target_recall: float
) -> float | None:
    eligible = precision[recall >= target_recall]
    return float(eligible.max()) if eligible.size else None


def _recall_at_precision(
    precision: np.ndarray, recall: np.ndarray, target_precision: float
) -> float | None:
    eligible = recall[precision >= target_precision]
    return float(eligible.max()) if eligible.size else None


def evaluate_probabilities(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    """Compute threshold-free and thresholded binary fraud metrics."""

    target_array = _binary_target(target)
    probability_array = _probabilities(probabilities, len(target_array))
    precision, recall, _ = precision_recall_curve(target_array, probability_array)
    return {
        "rows": int(len(target_array)),
        "positive_rows": int(target_array.sum()),
        "fraud_rate": float(target_array.mean()),
        "ranking": {
            "average_precision": float(
                average_precision_score(target_array, probability_array)
            ),
            "pr_auc_trapezoidal": float(
                np.trapezoid(precision[::-1], recall[::-1])
            ),
            "roc_auc": float(roc_auc_score(target_array, probability_array)),
        },
        "calibration": {
            "brier_score": float(brier_score_loss(target_array, probability_array)),
            "log_loss": float(
                log_loss(target_array, probability_array, labels=[0, 1])
            ),
        },
        "operating_point": _operating_metrics(
            target_array, probability_array, threshold
        ),
        "default_operating_point": _operating_metrics(
            target_array, probability_array, 0.5
        ),
        "precision_at_recall": {
            "0.80": _precision_at_recall(precision, recall, 0.80),
            "0.90": _precision_at_recall(precision, recall, 0.90),
            "0.95": _precision_at_recall(precision, recall, 0.95),
        },
        "recall_at_precision": {
            "0.50": _recall_at_precision(precision, recall, 0.50),
            "0.75": _recall_at_precision(precision, recall, 0.75),
            "0.90": _recall_at_precision(precision, recall, 0.90),
        },
    }


def threshold_selection_dict(selection: ThresholdSelection) -> dict[str, float | int]:
    """Return a JSON-compatible threshold selection mapping."""

    return asdict(selection)
