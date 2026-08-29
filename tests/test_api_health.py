from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_reports_loaded_frozen_model():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_version"]
    assert payload["calibration_method"] == "isotonic"
    assert payload["policy_status"] == "frozen"


def test_model_info_exposes_public_artifact_metadata_without_local_paths():
    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "LightGBM"
    assert payload["model_family"] == "lightgbm"
    assert payload["calibration_method"] == "isotonic"
    assert payload["raw_feature_count"] == 18
    assert payload["policy_status"] == "frozen"
    assert "C:\\" not in response.text
    assert "Users\\" not in response.text


def test_openapi_schema_is_generated():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/predict" in paths
    assert "/predict/batch" in paths


def test_predictor_instance_is_reused_across_requests():
    with TestClient(app) as client:
        predictor_id = id(client.app.state.predictor)
        client.get("/health")
        client.get("/model/info")

    assert id(app.state.predictor) == predictor_id
