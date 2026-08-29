from __future__ import annotations

import math

import pandas as pd
import pytest

from dashboard.utils import (
    DECISION_ORDER,
    batch_limit_from_openapi,
    batch_count,
    chunk_records,
    format_pd,
    format_score,
    merge_predictions,
    monitoring_summary,
    normalize_records_for_json,
    ordered_counts,
    portfolio_kpis,
    required_applicant_fields,
    distribution_comparison_frame,
    split_optional_labels,
    sorted_feature_drift,
    top_feature_psi,
    validate_portfolio_columns,
)


def test_format_pd_as_percentage():
    assert format_pd(0.03468) == "3.47%"


def test_format_score_as_whole_number():
    assert format_score(814.4888) == "814"


def test_normalize_records_converts_nan_to_none_and_numpy_scalars():
    records = normalize_records_for_json([{"a": float("nan"), "b": pd.Series([1]).iloc[0]}])

    assert records == [{"a": None, "b": 1}]


def test_chunk_records_uses_requested_chunk_size():
    records = [{"i": i} for i in range(5)]

    chunks = chunk_records(records, 2)

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_chunk_records_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        chunk_records([], 0)


def test_required_applicant_fields_from_openapi_schema():
    schema = {
        "components": {
            "schemas": {
                "ApplicantRequest": {
                    "required": ["person_age", "person_income"],
                    "properties": {
                        "person_age": {},
                        "person_income": {},
                        "other": {},
                    },
                }
            }
        }
    }

    assert required_applicant_fields(schema) == ["person_age", "person_income"]


def test_batch_limit_from_openapi_schema():
    schema = {
        "components": {
            "schemas": {
                "BatchPredictionRequest": {
                    "properties": {
                        "applicants": {
                            "maxItems": 250,
                        }
                    }
                }
            }
        }
    }

    assert batch_limit_from_openapi(schema) == 250


def test_validate_portfolio_columns_returns_missing_required_fields():
    df = pd.DataFrame({"person_age": [35]})

    assert validate_portfolio_columns(df, ["person_age", "person_income"]) == ["person_income"]


def test_portfolio_kpis_calculate_expected_rates():
    predictions = pd.DataFrame(
        {
            "pd": [0.1, 0.2, 0.3],
            "credit_score": [800, 700, 600],
            "decision": ["APPROVE", "MANUAL_REVIEW", "REJECT"],
        }
    )

    kpis = portfolio_kpis(predictions)

    assert kpis["total_applicants"] == 3
    assert math.isclose(kpis["mean_pd"], 0.2)
    assert math.isclose(kpis["approval_rate"], 1 / 3)
    assert math.isclose(kpis["manual_review_rate"], 1 / 3)
    assert math.isclose(kpis["rejection_rate"], 1 / 3)


def test_ordered_counts_uses_frozen_display_order():
    values = pd.Series(["REJECT", "APPROVE", "APPROVE"])

    counts = ordered_counts(values, DECISION_ORDER)

    assert counts.index.tolist() == ["APPROVE", "MANUAL_REVIEW", "REJECT"]
    assert counts.tolist() == [2, 0, 1]


def test_merge_predictions_preserves_input_row_order():
    inputs = pd.DataFrame({"row": ["a", "b"]})
    predictions = [
        {"pd": 0.2, "credit_score": 700, "risk_grade": "C", "decision": "MANUAL_REVIEW"},
        {"pd": 0.1, "credit_score": 800, "risk_grade": "B", "decision": "APPROVE"},
    ]

    merged = merge_predictions(inputs, predictions)

    assert merged["row"].tolist() == ["a", "b"]
    assert merged["pd"].tolist() == [0.2, 0.1]


def test_merge_predictions_rejects_count_mismatch():
    with pytest.raises(ValueError, match="Prediction count"):
        merge_predictions(pd.DataFrame({"row": ["a"]}), [])


def test_batch_count_handles_empty_and_non_empty_counts():
    assert batch_count(0, 1000) == 0
    assert batch_count(1001, 1000) == 2


def test_split_optional_labels_removes_target_from_applicant_frame():
    df = pd.DataFrame({"person_age": [35, 40], "loan_status": [0, 1], "client_ID": ["a", "b"]})

    applicants, labels = split_optional_labels(df, ["person_age"])

    assert applicants.columns.tolist() == ["person_age"]
    assert labels == [0, 1]


def test_monitoring_summary_counts_feature_statuses():
    report = {
        "monitoring_status": "WARNING",
        "current_rows": 2,
        "feature_drift": [{"status": "ALERT"}, {"status": "WARNING"}, {"status": "STABLE"}],
        "pd_drift": {"pd_psi": 0.1},
        "score_drift": {"score_psi": 0.2},
        "risk_grade_drift": {"risk_grade_psi": 0.3},
        "decision_drift": {"decision_psi": 0.4},
    }

    summary = monitoring_summary(report)

    assert summary["alert_feature_count"] == 1
    assert summary["warning_feature_count"] == 1
    assert summary["pd_psi"] == 0.1


def test_sorted_feature_drift_orders_by_severity_then_psi():
    rows = [
        {"feature": "stable", "status": "STABLE", "psi": 9.0},
        {"feature": "warning", "status": "WARNING", "psi": 0.2},
        {"feature": "alert", "status": "ALERT", "psi": 0.3},
    ]

    assert [row["feature"] for row in sorted_feature_drift(rows)] == ["alert", "warning", "stable"]


def test_top_feature_psi_returns_limited_dataframe():
    rows = [{"feature": f"f{i}", "status": "STABLE", "psi": float(i)} for i in range(5)]

    result = top_feature_psi(rows, n=2)

    assert result["feature"].tolist() == ["f4", "f3"]


def test_distribution_comparison_frame_shapes_reference_and_current():
    frame = distribution_comparison_frame(
        {
            "reference_distribution": {"A": 0.7, "B": 0.3},
            "current_distribution": {"A": 0.5, "B": 0.5},
        },
        "risk_grade",
    )

    assert frame.columns.tolist() == ["risk_grade", "reference_pct", "current_pct"]
    assert frame["current_pct"].tolist() == [0.5, 0.5]
