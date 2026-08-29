from __future__ import annotations

import numpy as np
import pytest

from src.models.scoring import CreditScoreConfig, pd_to_score, validate_score_monotonicity


@pytest.fixture()
def score_config(credit_policy):
    return CreditScoreConfig(**credit_policy["credit_score"])


def test_pd_to_score_outputs_configured_range(score_config):
    pd_values = np.array([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])

    scores = pd_to_score(pd_values, config=score_config)

    assert (scores >= score_config.score_min).all()
    assert (scores <= score_config.score_max).all()


def test_pd_to_score_is_monotonically_decreasing(score_config):
    pd_values = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 0.90])

    scores = pd_to_score(pd_values, config=score_config)

    assert (np.diff(scores) <= 0).all()
    assert validate_score_monotonicity(score_config)


def test_pd_to_score_handles_zero_and_one_with_epsilon_clipping(score_config):
    scores = pd_to_score(np.array([0, 1]), config=score_config)

    assert np.isfinite(scores).all()
    assert scores[0] == score_config.score_max
    assert scores[1] == score_config.score_min


@pytest.mark.parametrize("invalid_pd", [-0.01, 1.01, np.nan, np.inf])
def test_pd_to_score_rejects_invalid_pd_values(score_config, invalid_pd):
    with pytest.raises(ValueError):
        pd_to_score(np.array([invalid_pd]), config=score_config)
