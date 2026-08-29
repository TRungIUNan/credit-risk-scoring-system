from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def applicant_payload(small_valid_dataframe):
    return small_valid_dataframe.iloc[0].to_dict()


def fake_monitoring_report() -> dict:
    return {
        "monitoring_status": "STABLE",
        "reference_rows": 100,
        "current_rows": 1,
        "feature_drift": [],
        "pd_drift": {"pd_psi": 0.0, "pd_status": "STABLE"},
        "score_drift": {"score_psi": 0.0, "score_status": "STABLE"},
        "risk_grade_drift": {"risk_grade_psi": 0.0, "risk_grade_status": "STABLE"},
        "decision_drift": {"decision_psi": 0.0, "decision_status": "STABLE"},
        "performance": {"performance_status": "labels_unavailable"},
        "alerts": [],
    }


def test_monitoring_analyze_valid_drift_only_request(monkeypatch, api_client, applicant_payload):
    def fake_analyze(applicants, labels=None, predictor=None, reference_path=None, config=None):
        report = fake_monitoring_report()
        report["current_rows"] = len(applicants)
        report["performance"] = {"performance_status": "labels_unavailable"}
        return report

    monkeypatch.setattr("api.main.analyze_applicants", fake_analyze)

    response = api_client.post("/monitoring/analyze", json={"applicants": [applicant_payload]})

    assert response.status_code == 200
    assert response.json()["monitoring_status"] == "STABLE"
    assert response.json()["performance"]["performance_status"] == "labels_unavailable"


def test_monitoring_analyze_valid_labels_request(monkeypatch, api_client, applicant_payload):
    def fake_analyze(applicants, labels=None, predictor=None, reference_path=None, config=None):
        report = fake_monitoring_report()
        report["performance"] = {"performance_status": "available", "labeled_rows": len(labels)}
        return report

    monkeypatch.setattr("api.main.analyze_applicants", fake_analyze)

    response = api_client.post(
        "/monitoring/analyze",
        json={"applicants": [applicant_payload], "labels": [0]},
    )

    assert response.status_code == 200
    assert response.json()["performance"]["performance_status"] == "available"


def test_monitoring_analyze_labels_length_mismatch_returns_422(api_client, applicant_payload):
    response = api_client.post(
        "/monitoring/analyze",
        json={"applicants": [applicant_payload], "labels": [0, 1]},
    )

    assert response.status_code == 422


def test_monitoring_analyze_invalid_applicant_returns_422(api_client, applicant_payload):
    applicant_payload["loan_status"] = 0

    response = api_client.post("/monitoring/analyze", json={"applicants": [applicant_payload]})

    assert response.status_code == 422


def test_monitoring_analyze_empty_batch_returns_422(api_client):
    response = api_client.post("/monitoring/analyze", json={"applicants": []})

    assert response.status_code == 422


def test_monitoring_analyze_too_large_batch_returns_422(api_client, applicant_payload):
    response = api_client.post(
        "/monitoring/analyze",
        json={"applicants": [applicant_payload for _ in range(1001)]},
    )

    assert response.status_code == 422
