from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    MISSING_BUCKET,
    OTHER_BUCKET,
    build_categorical_reference,
    build_numeric_reference,
    categorical_distribution,
    compare_feature,
    numeric_distribution,
    population_stability_index,
)
from src.monitoring.monitor import (
    analyze_current_batch,
    build_reference_profile,
    load_reference_profile,
    performance_monitoring,
    save_json,
)


class FakePredictor:
    required_features = ["num", "cat"]
    feature_schema = {
        "schema_version": "fake-version",
        "raw_primary_features": ["num", "cat"],
        "numeric_features_after_engineering": ["num"],
    }
    credit_policy = {
        "policy_status": "frozen",
        "calibration_method": "isotonic",
        "risk_grade_labels": ["A", "B", "C", "D", "E"],
    }
    model_metadata = {
        "artifact_version": "fake-version",
        "metrics": {"locked_test": {"roc_auc": 0.8, "pr_auc": 0.7, "ks": 0.5, "gini": 0.6, "brier": 0.1}},
    }

    def predict_batch(self, applicants: pd.DataFrame) -> pd.DataFrame:
        pd_values = np.clip(pd.to_numeric(applicants["num"], errors="coerce").fillna(0) / 100, 0, 1)
        return pd.DataFrame(
            {
                "pd": pd_values,
                "credit_score": 850 - pd_values * 500,
                "risk_grade": np.where(pd_values < 0.2, "A", np.where(pd_values < 0.5, "C", "E")),
                "decision": np.where(pd_values < 0.2, "APPROVE", np.where(pd_values < 0.7, "MANUAL_REVIEW", "REJECT")),
            }
        )


def test_numeric_psi_increases_with_shift_strength():
    reference = build_numeric_reference(pd.Series(range(100)), n_bins=5)
    same = numeric_distribution(pd.Series(range(100)), reference["bin_edges"])
    mild = numeric_distribution(pd.Series(range(20, 120)), reference["bin_edges"])
    strong = numeric_distribution(pd.Series(range(100, 200)), reference["bin_edges"])

    ref_values = list(reference["proportions"].values())
    same_psi = population_stability_index(ref_values, list(same.values()))
    mild_psi = population_stability_index(ref_values, list(mild.values()))
    strong_psi = population_stability_index(ref_values, list(strong.values()))

    assert same_psi == 0
    assert mild_psi > same_psi
    assert strong_psi > mild_psi


def test_numeric_current_data_does_not_refit_reference_bin_edges():
    reference = build_numeric_reference(pd.Series(range(100)), n_bins=4)
    original_edges = list(reference["bin_edges"])

    compare_feature("num", reference, pd.Series(range(1000, 1100)), 1e-6, 0.1, 0.25)

    assert reference["bin_edges"] == original_edges


def test_categorical_distribution_maps_unseen_and_missing_buckets():
    reference = build_categorical_reference(pd.Series(["A", "A", "B", None]))
    current = categorical_distribution(pd.Series(["A", "C", None, "C"]), reference["categories"])

    assert OTHER_BUCKET in current
    assert MISSING_BUCKET in current
    assert math.isclose(current[OTHER_BUCKET], 0.5)
    assert math.isclose(current[MISSING_BUCKET], 0.25)


def test_missingness_delta_is_reported():
    reference = build_numeric_reference(pd.Series([1, 2, 3, 4, np.nan]), n_bins=2)

    result = compare_feature("num", reference, pd.Series([1, 2, np.nan, np.nan, np.nan]), 1e-6, 0.1, 0.25)

    assert math.isclose(result["reference_missing_rate"], 0.2)
    assert math.isclose(result["current_missing_rate"], 0.6)
    assert math.isclose(result["missing_rate_delta"], 0.4)


def test_build_reference_profile_contains_required_aggregate_keys(tmp_path):
    reference = pd.DataFrame({"num": [5, 15, 25, 75], "cat": ["x", "x", "y", "z"]})
    path = tmp_path / "reference_profile.json"

    profile = build_reference_profile(reference, config={"monitoring": {"numeric_bins": 2}}, predictor=FakePredictor(), output_path=path)
    loaded = load_reference_profile(path, config={"monitoring": {}})

    assert profile["reference_rows"] == 4
    assert loaded["model_version"] == "fake-version"
    assert set(loaded["features"]) == {"num", "cat"}
    text = path.read_text(encoding="utf-8")
    assert "raw_path" not in text
    assert str(tmp_path) not in text


def test_analyze_current_batch_outputs_all_monitoring_sections():
    reference = pd.DataFrame({"num": [5, 15, 25, 75] * 6, "cat": ["x", "x", "y", "z"] * 6})
    profile = build_reference_profile(reference, config={"monitoring": {"numeric_bins": 2}}, predictor=FakePredictor())
    current = pd.DataFrame({"num": [80, 90, 95, 99] * 6, "cat": ["new", None, "x", "new"] * 6})
    labels = [0, 1] * 12

    report = analyze_current_batch(
        current,
        labels=labels,
        reference_profile=profile,
        config={"monitoring": {"numeric_bins": 2, "min_labeled_rows": 4}},
        predictor=FakePredictor(),
    )

    assert report["monitoring_status"] in {"STABLE", "WARNING", "ALERT"}
    assert len(report["feature_drift"]) == 2
    assert "pd_psi" in report["pd_drift"]
    assert "score_psi" in report["score_drift"]
    assert "risk_grade_psi" in report["risk_grade_drift"]
    assert "decision_psi" in report["decision_drift"]
    assert report["performance"]["performance_status"] == "available"
    assert "calibration_gap" in report["performance"]


def test_performance_monitoring_label_guards():
    pd_values = pd.Series([0.1, 0.2, 0.8, 0.9])

    assert performance_monitoring(None, pd_values, {}, 2)["performance_status"] == "labels_unavailable"
    assert performance_monitoring([0, 1], pd_values, {}, 2)["performance_status"] == "invalid_labels"
    assert performance_monitoring([0, 0, 0, 0], pd_values, {}, 2)["performance_status"] == "invalid_labels"
    assert performance_monitoring([0, 1, 0, 1], pd_values, {}, 10)["performance_status"] == "insufficient_labels"


def test_save_json_uses_json_safe_infinity_tokens(tmp_path):
    path = tmp_path / "payload.json"

    save_json({"edges": [float("-inf"), 1.0, float("inf")]}, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"edges": ["__NEG_INF__", 1.0, "__POS_INF__"]}
