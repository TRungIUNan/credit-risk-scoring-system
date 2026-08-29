from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.models.evaluate import evaluate_binary_probabilities, ks_statistic


def test_binary_metrics_match_sklearn_on_toy_data():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    metrics = evaluate_binary_probabilities(y_true, y_proba)

    assert metrics["roc_auc"] == roc_auc_score(y_true, y_proba)
    assert metrics["pr_auc"] == average_precision_score(y_true, y_proba)
    assert metrics["brier"] == brier_score_loss(y_true, y_proba)
    assert metrics["gini"] == 2 * metrics["roc_auc"] - 1
    assert metrics["ks"] == 1.0


def test_ks_statistic_returns_expected_value_for_separated_scores():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


@pytest.mark.parametrize(
    ("y_true", "y_proba", "message"),
    [
        ([0, 1], [0.1], "same length"),
        ([0, 1], [0.1, np.nan], "NaN"),
        ([0, 1], [0.1, -0.1], "within"),
        ([0, 1], [0.1, 1.1], "within"),
        ([0, 2], [0.1, 0.9], "binary"),
        ([0, 0], [0.1, 0.2], "both binary classes"),
    ],
)
def test_binary_metrics_raise_clear_errors_for_invalid_inputs(y_true, y_proba, message):
    with pytest.raises(ValueError, match=message):
        evaluate_binary_probabilities(y_true, y_proba)
