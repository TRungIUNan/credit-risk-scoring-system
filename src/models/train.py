"""Train and freeze the notebook-selected credit-risk model artifacts.

Run from project root:
    python -m src.models.train
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.data.preprocessing import (
    BusinessRuleCleaner,
    CreditRiskFeatureBuilder,
    build_preprocessor,
    infer_feature_types,
)
from src.data.validation import load_dataset, run_validation
from src.decision.policy import (
    CreditPolicy,
    assign_credit_decision,
    assign_risk_grade,
    build_risk_grade_thresholds,
    calculate_decision_metrics,
)
from src.features.build_features import build_feature_decisions
from src.models.calibrate import fit_calibrated_pipeline, positive_class_proba
from src.models.evaluate import (
    evaluate_binary_probabilities,
    summarize_notebook_comparison,
)
from src.models.scoring import (
    CreditScoreConfig,
    pd_to_score,
    validate_score_monotonicity,
)


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if all((candidate / name).exists() for name in ("data", "src", "notebooks")):
            return candidate
    raise FileNotFoundError("Cannot find CREDIT_RISK_PROJECT root.")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else project_root / path


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def validate_raw_data(data_path: Path, sheet_name: str) -> dict[str, Any]:
    rows = load_dataset(data_path, sheet_name)
    results = run_validation(rows)
    failed = [result for result in results if result.status == "failed"]
    if failed:
        names = ", ".join(result.check_name for result in failed)
        raise RuntimeError(f"Raw data validation failed: {names}")
    return {
        "row_count": len(rows),
        "failed_checks": 0,
        "warning_or_watchlist_checks": sum(
            result.status in {"warning", "watchlist"} for result in results
        ),
        "checks": [result.__dict__ for result in results],
    }


def build_champion_pipeline(
    config: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    params = dict(config["model"]["params"])
    params.update(
        {
            "random_state": int(config["split"]["random_state"]),
            "objective": "binary",
            "verbosity": -1,
        }
    )
    model = LGBMClassifier(**params)
    return Pipeline(
        steps=[
            (
                "business_rules",
                BusinessRuleCleaner(
                    min_age=config["business_rules"]["min_age"],
                    max_age=config["business_rules"]["max_age"],
                    minimum_working_age=config["business_rules"]["minimum_working_age"],
                    original_missing_indicator_cols=tuple(
                        config["features"].get("original_missing_indicators", [])
                    ),
                ),
            ),
            ("feature_builder", CreditRiskFeatureBuilder()),
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    use_scaler=False,
                ),
            ),
            ("model", model),
        ]
    )


def run_smoke_checks(
    calibrated_model,
    X_sample: pd.DataFrame,
    raw_feature_cols: list[str],
    forbidden_features: list[str],
    score_config: CreditScoreConfig,
) -> dict[str, bool]:
    proba = calibrated_model.predict_proba(X_sample)
    pd_values = proba[:, 1]
    return {
        "predict_proba_shape_valid": bool(proba.shape == (len(X_sample), 2)),
        "probabilities_in_unit_interval": bool(np.all((pd_values >= 0.0) & (pd_values <= 1.0))),
        "score_monotonic_decreasing": validate_score_monotonicity(score_config),
        "no_forbidden_raw_features": not any(col in raw_feature_cols for col in forbidden_features),
    }


def train(config_path: Path) -> dict[str, Any]:
    project_root = find_project_root(Path.cwd().resolve())
    config = load_config(config_path)
    data_path = project_path(project_root, config["data"]["raw_path"])
    artifacts_dir = project_path(project_root, config["artifacts"]["output_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    validation_summary = validate_raw_data(data_path, config["data"]["sheet_name"])
    df = pd.read_excel(data_path, sheet_name=config["data"]["sheet_name"])

    target_col = config["data"]["target_col"]
    configured_primary = list(config["features"]["primary"])
    decisions = build_feature_decisions(df.columns)
    if configured_primary != decisions.primary_feature_cols:
        raise AssertionError(
            "Configured primary features do not match src.features.build_features decisions."
        )

    forbidden_features = (
        [target_col, config["data"]["id_col"]]
        + list(config["features"]["extended_only"])
        + list(config["features"]["audit_only"])
        + list(config["features"]["redundant_excluded"])
    )
    forbidden_present = [col for col in forbidden_features if col in configured_primary]
    if forbidden_present:
        raise AssertionError("Forbidden features in primary list: " + ", ".join(forbidden_present))

    X = df.loc[:, configured_primary]
    y = df[target_col].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["random_state"]),
        stratify=y,
        shuffle=True,
    )

    type_summary = infer_feature_types(X_dev, configured_primary)
    champion_pipeline = build_champion_pipeline(
        config,
        numeric_features=type_summary.numeric_features,
        categorical_features=type_summary.categorical_features,
    )
    champion_model = champion_pipeline.fit(X_dev, y_dev)
    calibrated_model = fit_calibrated_pipeline(
        champion_pipeline,
        X_dev,
        y_dev,
        method=config["calibration"]["method"],
        random_state=int(config["split"]["random_state"]),
    )

    dev_pd = positive_class_proba(calibrated_model, X_dev)
    test_pd = positive_class_proba(calibrated_model, X_test)
    dev_metrics = evaluate_binary_probabilities(y_dev, dev_pd)
    test_metrics = evaluate_binary_probabilities(y_test, test_pd)

    policy_cfg = CreditPolicy(
        approve_threshold=float(config["policy"]["approve_threshold"]),
        reject_threshold=float(config["policy"]["reject_threshold"]),
        risk_grade_labels=tuple(config["policy"]["risk_grade_labels"]),
        business_utility=dict(config["policy"]["business_utility"]),
    )
    grade_thresholds = build_risk_grade_thresholds(dev_pd)
    test_decisions = assign_credit_decision(
        test_pd,
        approve_threshold=policy_cfg.approve_threshold,
        reject_threshold=policy_cfg.reject_threshold,
    )
    test_grades = assign_risk_grade(
        test_pd,
        grade_thresholds=grade_thresholds,
        labels=policy_cfg.risk_grade_labels,
    )
    decision_metrics = calculate_decision_metrics(
        y_test,
        test_decisions,
        policy_cfg.business_utility,
    )

    score_config = CreditScoreConfig(**config["credit_score"])
    test_scores = pd_to_score(test_pd, config=score_config)
    notebook_comparison_df = summarize_notebook_comparison(
        test_metrics,
        config["notebook_reference"]["metrics"],
    )

    smoke_checks = run_smoke_checks(
        calibrated_model,
        X_test.head(25),
        configured_primary,
        forbidden_features,
        score_config,
    )
    if not all(smoke_checks.values()):
        failed = [name for name, passed in smoke_checks.items() if not passed]
        raise AssertionError("Smoke checks failed: " + ", ".join(failed))

    artifact_paths = {
        "champion_model": artifacts_dir / config["artifacts"]["champion_model"],
        "calibrated_model": artifacts_dir / config["artifacts"]["calibrated_model"],
        "feature_schema": artifacts_dir / config["artifacts"]["feature_schema"],
        "credit_policy": artifacts_dir / config["artifacts"]["credit_policy"],
        "model_metadata": artifacts_dir / config["artifacts"]["model_metadata"],
    }

    joblib.dump(champion_model, artifact_paths["champion_model"])
    joblib.dump(calibrated_model, artifact_paths["calibrated_model"])

    feature_schema = {
        "schema_version": config["project"]["artifact_version"],
        "target_col": target_col,
        "id_col": config["data"]["id_col"],
        "raw_primary_features": configured_primary,
        "numeric_features_after_engineering": type_summary.numeric_features,
        "categorical_features_after_engineering": type_summary.categorical_features,
        "engineered_features": type_summary.engineered_features,
        "extended_only_features": config["features"]["extended_only"],
        "audit_only_features": config["features"]["audit_only"],
        "redundant_excluded_features": config["features"]["redundant_excluded"],
        "original_missing_indicators": config["features"]["original_missing_indicators"],
        "feature_decision_note": decisions.feature_decision_note,
    }
    write_json(artifact_paths["feature_schema"], feature_schema)

    credit_policy = {
        "policy_status": config["policy"]["status"],
        "calibration_method": config["calibration"]["method"],
        "approve_threshold": policy_cfg.approve_threshold,
        "reject_threshold": policy_cfg.reject_threshold,
        "risk_grade_thresholds": grade_thresholds,
        "risk_grade_labels": list(policy_cfg.risk_grade_labels),
        "business_utility": policy_cfg.business_utility,
        "credit_score": score_config.to_dict(),
    }
    write_json(artifact_paths["credit_policy"], credit_policy)

    metadata = {
        "artifact_version": config["project"]["artifact_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": {
            "feature_notebook": "notebooks/03_feature_analysis.ipynb",
            "model_notebook": config["notebook_reference"]["source_notebook"],
            "note": "CLI training reuses frozen notebook decisions without retuning.",
        },
        "data": {
            "raw_path": config["data"]["raw_path"],
            "sheet_name": config["data"]["sheet_name"],
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "target_rate": float(y.mean()),
            "development_rows": int(len(X_dev)),
            "locked_test_rows": int(len(X_test)),
            "locked_test_share": float(len(X_test) / len(df)),
        },
        "validation_summary": validation_summary,
        "model": {
            "champion_name": config["model"]["champion_name"],
            "champion_key": config["model"]["champion_key"],
            "selected_params": config["model"]["params"],
            "random_state": config["split"]["random_state"],
            "cv_n_splits_reference": config["split"]["cv_n_splits"],
        },
        "calibration": {
            "method": config["calibration"]["method"],
            "final_artifact_fit": "Base model fitted on development split; isotonic calibrator fitted on development raw probabilities, matching notebook final-artifact flow.",
        },
        "metrics": {
            "development": dev_metrics,
            "locked_test": test_metrics,
            "locked_test_decision": decision_metrics,
            "locked_test_score_summary": {
                "min": float(np.min(test_scores)),
                "median": float(np.median(test_scores)),
                "max": float(np.max(test_scores)),
            },
            "locked_test_grade_counts": test_grades.value_counts(dropna=False).to_dict(),
            "notebook_vs_pipeline_comparison": notebook_comparison_df.to_dict(orient="records"),
        },
        "feature_schema": feature_schema,
        "credit_policy": credit_policy,
        "artifacts": {
            key: str(path.relative_to(project_root)).replace("\\", "/")
            for key, path in artifact_paths.items()
        },
        "smoke_checks": smoke_checks,
    }
    write_json(artifact_paths["model_metadata"], metadata)

    print("Frozen credit-risk training pipeline completed.")
    print(f"Champion model       : {config['model']['champion_name']}")
    print(f"Calibration method   : {config['calibration']['method']}")
    print(f"Raw primary features : {len(configured_primary)}")
    print(f"Engineered features  : {len(type_summary.engineered_features)}")
    print(f"Development rows     : {len(X_dev):,}")
    print(f"Locked Test rows     : {len(X_test):,}")
    print(f"Locked Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"Locked Test PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"Locked Test KS       : {test_metrics['ks']:.4f}")
    print(f"Locked Test Brier    : {test_metrics['brier']:.4f}")
    print(f"Approve threshold    : {policy_cfg.approve_threshold:.4f}")
    print(f"Reject threshold     : {policy_cfg.reject_threshold:.4f}")
    print("Artifacts:")
    for key, path in artifact_paths.items():
        print(f"- {key}: {path.relative_to(project_root)}")
    print("Smoke checks:")
    for check_name, passed in smoke_checks.items():
        print(f"- {check_name}: {'PASS' if passed else 'FAIL'}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen credit-risk artifacts.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config. Defaults to configs/config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root(Path.cwd().resolve())
    config_path = project_path(project_root, args.config)
    train(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
