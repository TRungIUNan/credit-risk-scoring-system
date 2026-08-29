"""Evaluation helpers for the frozen credit-risk model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_binary_probability_inputs(y_true, y_proba) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and probability estimates before metric calculation."""

    y_true_array = np.asarray(y_true)
    y_proba_array = np.asarray(y_proba, dtype=float)

    if y_true_array.shape[0] != y_proba_array.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")
    if y_true_array.ndim != 1 or y_proba_array.ndim != 1:
        raise ValueError("y_true and y_proba must be one-dimensional.")
    if not np.isfinite(y_proba_array).all():
        raise ValueError("y_proba must not contain NaN or infinite values.")
    if ((y_proba_array < 0.0) | (y_proba_array > 1.0)).any():
        raise ValueError("y_proba values must be within [0, 1].")

    unique_targets = set(pd.Series(y_true_array).dropna().unique().tolist())
    if unique_targets - {0, 1, "0", "1"}:
        raise ValueError("y_true must be binary with values 0/1.")
    if pd.Series(y_true_array).isna().any():
        raise ValueError("y_true must not contain missing values.")

    y_true_array = y_true_array.astype(int)
    if set(np.unique(y_true_array)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes 0 and 1.")

    return y_true_array, y_proba_array


def ks_statistic(y_true, y_proba) -> float:
    """Compute the binary Kolmogorov-Smirnov statistic."""

    y_true, y_proba = validate_binary_probability_inputs(y_true, y_proba)
    positives = y_proba[y_true == 1]
    negatives = y_proba[y_true == 0]
    thresholds = np.sort(np.unique(y_proba))
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    pos_cdf = np.searchsorted(np.sort(positives), thresholds, side="right") / len(positives)
    neg_cdf = np.searchsorted(np.sort(negatives), thresholds, side="right") / len(negatives)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def evaluate_binary_probabilities(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return standard probability and threshold metrics."""

    y_true, y_proba = validate_binary_probability_inputs(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, y_proba)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks": ks_statistic(y_true, y_proba),
        "gini": float(2 * roc_auc - 1),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def summarize_notebook_comparison(
    pipeline_metrics: dict[str, float],
    reference_metrics: dict[str, float],
) -> pd.DataFrame:
    """Compare frozen notebook metrics with the CLI pipeline run."""

    rows = []
    for metric, reference_value in reference_metrics.items():
        pipeline_value = pipeline_metrics.get(metric)
        rows.append(
            {
                "metric": metric,
                "notebook_value": reference_value,
                "pipeline_value": pipeline_value,
                "absolute_delta": (
                    None if pipeline_value is None else abs(pipeline_value - reference_value)
                ),
            }
        )
    return pd.DataFrame(rows)
