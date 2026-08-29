from __future__ import annotations

import joblib
import numpy as np
import pytest


@pytest.fixture(scope="module")
def calibrated_model(project_root):
    return joblib.load(project_root / "models" / "calibrated_model.joblib")


@pytest.mark.integration
def test_calibrated_artifact_loads(calibrated_model):
    assert calibrated_model.calibration_method == "isotonic"


@pytest.mark.integration
def test_predict_proba_contract(calibrated_model, sample_applicants):
    probabilities = calibrated_model.predict_proba(sample_applicants)

    assert probabilities.shape == (len(sample_applicants), 2)


def test_calibrated_probabilities_are_valid(calibrated_model, sample_applicants):
    probabilities = calibrated_model.predict_proba(sample_applicants)

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_calibrated_probability_rows_sum_to_one(calibrated_model, sample_applicants):
    probabilities = calibrated_model.predict_proba(sample_applicants)

    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(sample_applicants)))


def test_calibrated_artifact_predictions_are_deterministic(calibrated_model, sample_applicants):
    first = calibrated_model.predict_proba(sample_applicants)
    second = calibrated_model.predict_proba(sample_applicants)

    np.testing.assert_allclose(first, second)
