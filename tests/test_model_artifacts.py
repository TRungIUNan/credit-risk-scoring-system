from __future__ import annotations

import json
import re

import joblib
import pytest


REQUIRED_ARTIFACTS = [
    "champion_model.joblib",
    "calibrated_model.joblib",
    "feature_schema.json",
    "credit_policy.json",
    "model_metadata.json",
]


def test_required_model_artifacts_exist(project_root):
    for artifact_name in REQUIRED_ARTIFACTS:
        assert (project_root / "models" / artifact_name).exists()


def test_json_artifacts_parse_and_do_not_contain_absolute_local_paths(project_root):
    local_path_pattern = re.compile(r"C:\\|Users\\|Desktop\\")
    for artifact_name in ["feature_schema.json", "credit_policy.json", "model_metadata.json"]:
        text = (project_root / "models" / artifact_name).read_text(encoding="utf-8")
        json.loads(text)
        assert not local_path_pattern.search(text)


def test_feature_schema_matches_frozen_primary_contract(config, feature_schema):
    assert feature_schema["target_col"] == config["data"]["target_col"]
    assert feature_schema["id_col"] == config["data"]["id_col"]
    assert feature_schema["raw_primary_features"] == config["features"]["primary"]
    assert len(feature_schema["raw_primary_features"]) == len(config["features"]["primary"])


def test_feature_schema_excludes_leakage_and_audit_features(config, feature_schema):
    forbidden = {
        config["data"]["id_col"],
        config["data"]["target_col"],
        *config["features"]["extended_only"],
        *config["features"]["audit_only"],
    }

    assert forbidden.isdisjoint(feature_schema["raw_primary_features"])


def test_credit_policy_artifact_matches_config(config, credit_policy):
    assert credit_policy["approve_threshold"] == config["policy"]["approve_threshold"]
    assert credit_policy["reject_threshold"] == config["policy"]["reject_threshold"]
    assert credit_policy["policy_status"] == config["policy"]["status"]
    assert credit_policy["calibration_method"] == config["calibration"]["method"]
    assert credit_policy["risk_grade_labels"] == config["policy"]["risk_grade_labels"]
    assert credit_policy["credit_score"] == config["credit_score"]


def test_metadata_contains_required_model_contract(model_metadata):
    assert model_metadata["model"]["champion_name"] == "LightGBM"
    assert model_metadata["calibration"]["method"] == "isotonic"
    assert model_metadata["model"]["selected_params"]
    assert model_metadata["metrics"]["locked_test"]
    assert model_metadata["artifacts"]["calibrated_model"] == "models/calibrated_model.joblib"


@pytest.mark.integration
def test_joblib_model_artifacts_load(project_root):
    champion = joblib.load(project_root / "models" / "champion_model.joblib")
    calibrated = joblib.load(project_root / "models" / "calibrated_model.joblib")

    assert hasattr(champion, "predict_proba")
    assert hasattr(calibrated, "predict_proba")
