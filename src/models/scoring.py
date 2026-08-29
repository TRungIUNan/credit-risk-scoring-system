"""Credit score transformation for calibrated probability of default."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CreditScoreConfig:
    score_min: int = 300
    score_max: int = 850
    base_score: int = 575
    factor: float = 72.0
    pd_clip_epsilon: float = 1e-6

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def pd_to_score(probability_default, config: CreditScoreConfig | None = None) -> np.ndarray:
    """Convert calibrated PD to a clipped credit score."""

    cfg = config or CreditScoreConfig()
    pd_values = np.asarray(probability_default, dtype=float)
    if not np.isfinite(pd_values).all():
        raise ValueError("probability_default must not contain NaN or infinite values.")
    if ((pd_values < 0.0) | (pd_values > 1.0)).any():
        raise ValueError("probability_default values must be within [0, 1].")
    clipped_pd = np.clip(pd_values, cfg.pd_clip_epsilon, 1.0 - cfg.pd_clip_epsilon)
    odds_good_bad = (1.0 - clipped_pd) / clipped_pd
    scores = cfg.base_score + cfg.factor * np.log(odds_good_bad)
    return np.clip(scores, cfg.score_min, cfg.score_max)


def validate_score_monotonicity(config: CreditScoreConfig | None = None) -> bool:
    """Verify that higher PD never maps to a higher score."""

    pd_grid = np.linspace(0.001, 0.999, 999)
    scores = pd_to_score(pd_grid, config=config)
    return bool(np.all(np.diff(scores) <= 1e-12))
