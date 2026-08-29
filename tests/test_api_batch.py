from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import MAX_BATCH_SIZE


@pytest.fixture()
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def applicant_records(small_valid_dataframe):
    frame = small_valid_dataframe.astype(object).where(small_valid_dataframe.notna(), None)
    return frame.to_dict(orient="records")


def test_predict_batch_valid_request_preserves_order_and_count(api_client, applicant_records):
    response = api_client.post("/predict/batch", json={"applicants": applicant_records[:3]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert len(payload["predictions"]) == 3
    assert [prediction["pd"] for prediction in payload["predictions"]]


def test_predict_batch_each_prediction_is_valid(api_client, applicant_records, credit_policy):
    response = api_client.post("/predict/batch", json={"applicants": applicant_records[:2]})

    assert response.status_code == 200
    for prediction in response.json()["predictions"]:
        assert 0.0 <= prediction["pd"] <= 1.0
        assert credit_policy["credit_score"]["score_min"] <= prediction["credit_score"] <= credit_policy["credit_score"]["score_max"]
        assert prediction["risk_grade"] in credit_policy["risk_grade_labels"]
        assert prediction["decision"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}


def test_predict_batch_empty_list_returns_422(api_client):
    response = api_client.post("/predict/batch", json={"applicants": []})

    assert response.status_code == 422


def test_predict_batch_too_large_returns_422(api_client, applicant_records):
    records = [applicant_records[0] for _ in range(MAX_BATCH_SIZE + 1)]

    response = api_client.post("/predict/batch", json={"applicants": records})

    assert response.status_code == 422


def test_predict_batch_rejects_whole_request_when_one_applicant_invalid(api_client, applicant_records):
    records = applicant_records[:2]
    records[1] = records[1].copy()
    records[1]["loan_grade"] = "A"

    response = api_client.post("/predict/batch", json={"applicants": records})

    assert response.status_code == 422
