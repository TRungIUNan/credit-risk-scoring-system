from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import infer_feature_types
from src.models.train import build_champion_pipeline, load_config


def test_training_config_loads(project_root, config):
    loaded = load_config(project_root / "configs" / "config.yaml")

    assert loaded == config


def test_training_pipeline_can_be_constructed_without_fitting_full_model(config, small_valid_dataframe):
    type_summary = infer_feature_types(small_valid_dataframe, config["features"]["primary"])

    pipeline = build_champion_pipeline(
        config,
        numeric_features=type_summary.numeric_features,
        categorical_features=type_summary.categorical_features,
    )

    assert [name for name, _ in pipeline.steps] == [
        "business_rules",
        "feature_builder",
        "preprocessor",
        "model",
    ]


def test_feature_order_is_stable_between_config_and_schema(config, feature_schema):
    assert config["features"]["primary"] == feature_schema["raw_primary_features"]


@pytest.mark.integration
def test_end_to_end_inference_smoke(project_root, feature_schema, credit_policy, sample_applicants):
    import joblib

    from src.decision.policy import assign_credit_decision, assign_risk_grade
    from src.models.scoring import CreditScoreConfig, pd_to_score

    model = joblib.load(project_root / "models" / "calibrated_model.joblib")
    applicants = sample_applicants.loc[:, feature_schema["raw_primary_features"]]

    pd_values = model.predict_proba(applicants)[:, 1]
    scores = pd_to_score(pd_values, CreditScoreConfig(**credit_policy["credit_score"]))
    grades = assign_risk_grade(
        pd_values,
        credit_policy["risk_grade_thresholds"],
        tuple(credit_policy["risk_grade_labels"]),
    )
    decisions = assign_credit_decision(
        pd_values,
        credit_policy["approve_threshold"],
        credit_policy["reject_threshold"],
    )

    assert np.isfinite(pd_values).all()
    assert ((pd_values >= 0.0) & (pd_values <= 1.0)).all()
    assert ((scores >= credit_policy["credit_score"]["score_min"]) & (scores <= credit_policy["credit_score"]["score_max"])).all()
    assert not pd.isna(grades).any()
    assert set(decisions).issubset({"APPROVE", "MANUAL_REVIEW", "REJECT"})
