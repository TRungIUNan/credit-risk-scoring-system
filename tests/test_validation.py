from __future__ import annotations

import copy

from src.data.validation import run_validation
from src.features.build_features import build_feature_decisions


def result_by_name(results, name):
    return next(result for result in results if result.check_name == name)


def test_valid_schema_passes(full_schema_rows):
    results = run_validation(full_schema_rows)

    assert result_by_name(results, "schema_expected_columns").status == "passed"
    assert result_by_name(results, "required_column_loan_status").status == "passed"
    assert result_by_name(results, "required_column_client_ID").status == "passed"


def test_missing_required_column_is_reported(full_schema_rows):
    rows = [{k: v for k, v in row.items() if k != "loan_status"} for row in full_schema_rows]

    results = run_validation(rows)

    assert result_by_name(results, "schema_expected_columns").status == "failed"
    assert result_by_name(results, "required_column_loan_status").status == "failed"


def test_non_binary_target_fails(full_schema_rows):
    rows = copy.deepcopy(full_schema_rows)
    rows[0]["loan_status"] = 2

    results = run_validation(rows)

    target_result = result_by_name(results, "target_domain")
    assert target_result.status == "failed"
    assert target_result.affected_rows == 1


def test_missing_target_fails(full_schema_rows):
    rows = copy.deepcopy(full_schema_rows)
    rows[0]["loan_status"] = None

    results = run_validation(rows)

    assert result_by_name(results, "target_domain").status == "failed"


def test_duplicate_client_id_is_detected(full_schema_rows):
    rows = copy.deepcopy(full_schema_rows)
    rows[1]["client_ID"] = rows[0]["client_ID"]

    results = run_validation(rows)

    duplicate_result = result_by_name(results, "duplicate_client_id")
    assert duplicate_result.status == "warning"
    assert duplicate_result.affected_rows == 2


def test_primary_feature_decisions_exclude_forbidden_modeling_columns(full_schema_rows):
    forbidden = {
        "client_ID",
        "loan_status",
        "loan_grade",
        "loan_int_rate",
        "gender",
        "country",
        "state",
        "city",
        "city_latitude",
        "city_longitude",
    }

    decisions = build_feature_decisions(full_schema_rows[0].keys())

    assert forbidden.isdisjoint(decisions.primary_feature_cols)


def test_validation_audits_missing_without_filling_values(full_schema_rows):
    rows = copy.deepcopy(full_schema_rows)
    rows[0]["person_emp_length"] = None

    before = copy.deepcopy(rows)
    results = run_validation(rows)

    assert rows == before
    assert result_by_name(results, "expected_missing_person_emp_length").affected_rows == 1
