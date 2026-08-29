from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.inference.predictor import CreditRiskPredictor


@pytest.fixture(scope="module")
def predictor():
    return CreditRiskPredictor()


@pytest.fixture()
def applicant(small_valid_dataframe):
    return small_valid_dataframe.iloc[0].to_dict()


def test_predictor_loads_frozen_artifacts(predictor):
    assert predictor.calibrated_model is not None
    assert predictor.feature_schema["raw_primary_features"]
    assert predictor.credit_policy["policy_status"] == "frozen"
    assert predictor.credit_policy["calibration_method"] == predictor.model_metadata["calibration"]["method"]


def test_valid_single_applicant_predicts_required_output(predictor, applicant):
    result = predictor.predict(applicant)

    assert {"pd", "credit_score", "risk_grade", "decision"}.issubset(result)
    assert result["risk_drivers"] is None
    assert result["explanation_status"] == "explanation_unavailable"


def test_single_prediction_values_are_valid(predictor, applicant, credit_policy):
    result = predictor.predict(applicant)

    assert 0.0 <= result["pd"] <= 1.0
    assert credit_policy["credit_score"]["score_min"] <= result["credit_score"] <= credit_policy["credit_score"]["score_max"]
    assert result["risk_grade"] in credit_policy["risk_grade_labels"]
    assert result["decision"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}


def test_same_input_returns_same_result(predictor, applicant):
    first = predictor.predict(applicant)
    second = predictor.predict(applicant)

    assert first == second


def test_missing_required_feature_raises_clear_error(predictor, applicant):
    applicant.pop("person_income")

    with pytest.raises(ValueError, match="Missing required applicant fields: person_income"):
        predictor.predict(applicant)


@pytest.mark.parametrize("field", ["loan_grade", "loan_int_rate", "loan_status", "client_ID"])
def test_forbidden_leakage_or_target_field_is_rejected(predictor, applicant, field):
    applicant[field] = "forbidden"

    with pytest.raises(ValueError, match="Forbidden applicant fields"):
        predictor.predict(applicant)


@pytest.mark.parametrize("field", ["gender", "country", "state", "city", "city_latitude", "city_longitude"])
def test_audit_only_fields_are_rejected(predictor, applicant, field):
    applicant[field] = "audit-only"

    with pytest.raises(ValueError, match="Forbidden applicant fields"):
        predictor.predict(applicant)


def test_non_forbidden_extra_field_is_ignored_and_reported(predictor, applicant):
    applicant["campaign_source"] = "branch"

    result = predictor.predict(applicant)

    assert result["ignored_extra_fields"] == ["campaign_source"]


def test_input_dict_is_not_mutated(predictor, applicant):
    original = copy.deepcopy(applicant)

    predictor.predict(applicant)

    assert applicant == original


def test_missing_person_emp_length_is_supported_by_frozen_pipeline(predictor, applicant):
    applicant["person_emp_length"] = None

    result = predictor.predict(applicant)

    assert 0.0 <= result["pd"] <= 1.0
    assert result["decision"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}


def test_invalid_business_values_are_handled_by_frozen_pipeline(predictor, applicant):
    applicant["person_age"] = 144
    applicant["person_emp_length"] = 25

    result = predictor.predict(applicant)

    assert 0.0 <= result["pd"] <= 1.0
    assert np.isfinite(result["credit_score"])


def test_predict_accepts_one_row_dataframe(predictor, small_valid_dataframe):
    result = predictor.predict(small_valid_dataframe.head(1))

    assert 0.0 <= result["pd"] <= 1.0


def test_predict_rejects_multirow_dataframe(predictor, small_valid_dataframe):
    with pytest.raises(ValueError, match="exactly one applicant row"):
        predictor.predict(small_valid_dataframe.head(2))


def test_batch_prediction_works_and_preserves_index(predictor, small_valid_dataframe, credit_policy):
    batch = small_valid_dataframe.copy()
    batch.index = ["a", "b", "c", "d"]

    result = predictor.predict_batch(batch)

    assert list(result.index) == ["a", "b", "c", "d"]
    assert list(result.columns) == [
        "pd",
        "credit_score",
        "risk_grade",
        "decision",
        "risk_drivers",
        "explanation_status",
        "model_version",
        "calibration_method",
        "policy_status",
        "ignored_extra_fields",
    ]
    assert ((result["pd"] >= 0.0) & (result["pd"] <= 1.0)).all()
    assert set(result["risk_grade"]).issubset(set(credit_policy["risk_grade_labels"]))
    assert set(result["decision"]).issubset({"APPROVE", "MANUAL_REVIEW", "REJECT"})


def test_batch_prediction_rejects_empty_dataframe(predictor, small_valid_dataframe):
    with pytest.raises(ValueError, match="at least one applicant"):
        predictor.predict_batch(small_valid_dataframe.iloc[0:0])


def test_feature_order_alignment_is_schema_order(predictor, small_valid_dataframe, feature_schema):
    shuffled = small_valid_dataframe.loc[:, list(reversed(feature_schema["raw_primary_features"]))]

    direct = predictor.predict_batch(small_valid_dataframe)
    shuffled_result = predictor.predict_batch(shuffled)

    pd.testing.assert_series_equal(direct["pd"], shuffled_result["pd"])


def test_model_is_loaded_once_per_predictor_instance(predictor, applicant):
    model_id = id(predictor.calibrated_model)

    predictor.predict(applicant)
    predictor.predict(applicant)

    assert id(predictor.calibrated_model) == model_id


def test_decision_is_consistent_with_returned_pd(predictor, applicant, credit_policy):
    result = predictor.predict(applicant)
    pd_value = result["pd"]

    if pd_value < credit_policy["approve_threshold"]:
        assert result["decision"] == "APPROVE"
    elif pd_value >= credit_policy["reject_threshold"]:
        assert result["decision"] == "REJECT"
    else:
        assert result["decision"] == "MANUAL_REVIEW"


def test_golden_samples_keep_score_order_consistent_with_pd(predictor, small_valid_dataframe):
    golden = small_valid_dataframe.copy()
    golden.loc[0, ["loan_to_income_ratio", "debt_to_income_ratio", "credit_utilization_ratio"]] = [0.05, 0.10, 0.10]
    golden.loc[1, ["loan_to_income_ratio", "debt_to_income_ratio", "credit_utilization_ratio"]] = [0.25, 0.35, 0.60]
    golden.loc[2, ["loan_to_income_ratio", "debt_to_income_ratio", "credit_utilization_ratio"]] = [0.75, 0.90, 0.95]

    result = predictor.predict_batch(golden.iloc[:3])
    ordered = result.sort_values("pd")

    assert (ordered["credit_score"].diff().dropna() <= 0).all()


def test_wrong_artifact_directory_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing required inference artifacts"):
        CreditRiskPredictor(artifact_dir=tmp_path)
