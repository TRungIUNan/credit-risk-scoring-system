from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.inference.predictor import CreditRiskPredictor


@pytest.fixture()
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def applicant_payload(small_valid_dataframe):
    return small_valid_dataframe.iloc[0].to_dict()


def test_predict_valid_applicant_returns_credit_risk_outputs(api_client, applicant_payload, credit_policy):
    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 200
    payload = response.json()
    assert {"pd", "credit_score", "risk_grade", "decision"}.issubset(payload)
    assert 0.0 <= payload["pd"] <= 1.0
    assert credit_policy["credit_score"]["score_min"] <= payload["credit_score"] <= credit_policy["credit_score"]["score_max"]
    assert payload["risk_grade"] in credit_policy["risk_grade_labels"]
    assert payload["decision"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}


def test_predict_allows_missing_employment_length(api_client, applicant_payload):
    applicant_payload["person_emp_length"] = None

    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 200
    assert 0.0 <= response.json()["pd"] <= 1.0


def test_predict_is_deterministic(api_client, applicant_payload):
    first = api_client.post("/predict", json=applicant_payload).json()
    second = api_client.post("/predict", json=applicant_payload).json()

    assert first == second


def test_predict_result_matches_direct_predictor(api_client, applicant_payload):
    direct = CreditRiskPredictor().predict(applicant_payload)
    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["pd"] == direct["pd"]
    assert payload["credit_score"] == direct["credit_score"]
    assert payload["risk_grade"] == direct["risk_grade"]
    assert payload["decision"] == direct["decision"]


def test_predict_missing_required_field_returns_422(api_client, applicant_payload):
    applicant_payload.pop("person_income")

    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["loan_grade", "loan_int_rate", "loan_status", "client_ID"])
def test_predict_forbidden_leakage_or_target_field_returns_422(api_client, applicant_payload, field):
    applicant_payload[field] = "forbidden"

    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["gender", "country", "state", "city", "city_latitude", "city_longitude"])
def test_predict_audit_only_field_returns_422(api_client, applicant_payload, field):
    applicant_payload[field] = "audit-only"

    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 422


def test_predict_wrong_obvious_type_returns_422(api_client, applicant_payload):
    applicant_payload["person_income"] = "not-a-number"

    response = api_client.post("/predict", json=applicant_payload)

    assert response.status_code == 422


def test_predict_business_rule_invalid_values_flow_through_pipeline(api_client, applicant_payload):
    payload = copy.deepcopy(applicant_payload)
    payload["person_age"] = 144
    payload["person_emp_length"] = 25

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 0.0 <= response.json()["pd"] <= 1.0
