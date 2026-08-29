from __future__ import annotations

import json

import pytest
import requests

from dashboard.api_client import (
    CreditRiskAPIClient,
    CreditRiskAPIError,
    CreditRiskAPIUnavailable,
)


class FakeSession:
    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        self.calls = []

    def request(self, method, url, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )
        if self.exception:
            raise self.exception
        return self.response


def make_response(status_code: int, payload) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


def test_health_successful_request():
    session = FakeSession(make_response(200, {"status": "healthy"}))
    client = CreditRiskAPIClient(base_url="http://api", session=session, timeout=3)

    payload = client.health()

    assert payload == {"status": "healthy"}
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"] == "http://api/health"
    assert session.calls[0]["timeout"] == 3


def test_model_info_successful_request():
    session = FakeSession(make_response(200, {"model_name": "LightGBM"}))
    client = CreditRiskAPIClient(base_url="http://api/", session=session)

    assert client.model_info()["model_name"] == "LightGBM"
    assert session.calls[0]["url"] == "http://api/model/info"


def test_predict_successful_request_serializes_applicant():
    session = FakeSession(make_response(200, {"pd": 0.1}))
    client = CreditRiskAPIClient(base_url="http://api", session=session)
    applicant = {"person_age": 35}

    assert client.predict(applicant) == {"pd": 0.1}
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["json"] == applicant


def test_batch_predict_successful_request_serializes_applicants():
    session = FakeSession(make_response(200, {"count": 1, "predictions": [{"pd": 0.1}]}))
    client = CreditRiskAPIClient(base_url="http://api", session=session)

    payload = client.predict_batch([{"person_age": 35}])

    assert payload["count"] == 1
    assert session.calls[0]["json"] == {"applicants": [{"person_age": 35}]}


def test_monitoring_reference_successful_request():
    session = FakeSession(make_response(200, {"reference_status": "available"}))
    client = CreditRiskAPIClient(base_url="http://api", session=session)

    assert client.monitoring_reference()["reference_status"] == "available"
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"] == "http://api/monitoring/reference"


def test_monitoring_analyze_serializes_optional_labels():
    session = FakeSession(make_response(200, {"monitoring_status": "STABLE"}))
    client = CreditRiskAPIClient(base_url="http://api", session=session)

    payload = client.monitoring_analyze([{"person_age": 35}], labels=[0])

    assert payload["monitoring_status"] == "STABLE"
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"] == "http://api/monitoring/analyze"
    assert session.calls[0]["json"] == {"applicants": [{"person_age": 35}], "labels": [0]}


def test_timeout_raises_unavailable_error():
    client = CreditRiskAPIClient(
        base_url="http://api",
        session=FakeSession(exception=requests.Timeout()),
    )

    with pytest.raises(CreditRiskAPIUnavailable, match="timed out"):
        client.health()


def test_connection_error_raises_unavailable_error():
    client = CreditRiskAPIClient(
        base_url="http://api",
        session=FakeSession(exception=requests.ConnectionError()),
    )

    with pytest.raises(CreditRiskAPIUnavailable, match="currently unavailable"):
        client.health()


def test_http_422_returns_meaningful_client_error():
    response = make_response(
        422,
        {"detail": [{"loc": ["body", "person_income"], "msg": "Field required"}]},
    )
    client = CreditRiskAPIClient(base_url="http://api", session=FakeSession(response))

    with pytest.raises(CreditRiskAPIError, match="body.person_income: Field required") as exc_info:
        client.predict({})

    assert exc_info.value.status_code == 422


def test_http_503_returns_model_unavailable_error():
    response = make_response(503, {"detail": "Credit risk predictor is not available."})
    client = CreditRiskAPIClient(base_url="http://api", session=FakeSession(response))

    with pytest.raises(CreditRiskAPIError, match="predictor is not available") as exc_info:
        client.health()

    assert exc_info.value.status_code == 503
