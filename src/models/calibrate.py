"""Probability calibration helpers for credit-risk models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def positive_class_proba(model, X) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""

    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("predict_proba must return two-class probabilities.")
    return np.asarray(probabilities[:, 1], dtype=float)


class ProbabilityCalibratedPipeline(BaseEstimator, ClassifierMixin):
    """Wrap a fitted classifier and a fitted one-dimensional calibrator."""

    def __init__(self, base_pipeline, calibration_method: str, calibrator):
        self.base_pipeline = base_pipeline
        self.calibration_method = calibration_method
        self.calibrator = calibrator

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw_pd = positive_class_proba(self.base_pipeline, X)
        calibrated_pd = np.asarray(self.calibrator.predict(raw_pd), dtype=float)
        calibrated_pd = np.clip(calibrated_pd, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_pd, calibrated_pd])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_probability_calibrator(
    raw_probabilities,
    y_true,
    method: str,
    random_state: int = 42,
):
    """Fit the frozen notebook calibration method."""

    raw_probabilities = np.asarray(raw_probabilities, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    if method == "isotonic":
        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        return calibrator.fit(raw_probabilities, y_true)
    if method in {"sigmoid", "platt"}:
        calibrator = LogisticRegression(random_state=random_state)
        return calibrator.fit(raw_probabilities.reshape(-1, 1), y_true)
    if method in {"none", "raw"}:
        return None
    raise ValueError(f"Unsupported calibration method: {method}")


def fit_calibrated_pipeline(
    base_pipeline,
    X_train,
    y_train,
    method: str = "isotonic",
    random_state: int = 42,
) -> ProbabilityCalibratedPipeline:
    """Fit a base pipeline and calibrator using the notebook's final-artifact flow."""

    fitted_base = clone(base_pipeline).fit(X_train, y_train)
    raw_probabilities = positive_class_proba(fitted_base, X_train)
    calibrator = fit_probability_calibrator(
        raw_probabilities,
        y_train,
        method=method,
        random_state=random_state,
    )
    if calibrator is None:
        calibrator = _IdentityCalibrator()
    return ProbabilityCalibratedPipeline(
        base_pipeline=fitted_base,
        calibration_method=method,
        calibrator=calibrator,
    ).fit(None, None)


class _IdentityCalibrator:
    def predict(self, raw_probabilities):
        return np.asarray(raw_probabilities, dtype=float)
