"""Monitoring orchestration and CLI for the frozen credit-risk system."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.inference.predictor import CreditRiskPredictor
from src.models.evaluate import evaluate_binary_probabilities
from src.monitoring.drift import (
    build_categorical_reference,
    build_numeric_reference,
    compare_categorical,
    compare_feature,
    compare_numeric,
    overall_status,
    sort_drift_rows,
)


DECISION_CATEGORIES = ["APPROVE", "MANUAL_REVIEW", "REJECT"]
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def monitoring_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "output_dir": "reports/monitoring",
        "reference_profile": "reference_profile.json",
        "numeric_bins": 10,
        "epsilon": 1e-6,
        "psi_warning_threshold": 0.10,
        "psi_alert_threshold": 0.25,
        "min_labeled_rows": 20,
        "max_batch_size": 1000,
    }
    settings.update(config.get("monitoring", {}))
    return settings


def reference_profile_path(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    settings = monitoring_settings(config)
    directory = Path(output_dir or settings["output_dir"])
    return directory / settings["reference_profile"]


def recreate_development_population(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the frozen Development split without retraining the model."""

    data_cfg = config["data"]
    split_cfg = config["split"]
    features = list(config["features"]["primary"])
    df = pd.read_excel(data_cfg["raw_path"], sheet_name=data_cfg["sheet_name"])
    X = df.loc[:, features]
    y = df[data_cfg["target_col"]].astype(int)
    stratify = y if bool(split_cfg.get("stratify", True)) else None
    X_dev, _X_test, y_dev, _y_test = train_test_split(
        X,
        y,
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["random_state"]),
        stratify=stratify,
        shuffle=True,
    )
    return X_dev.reset_index(drop=True), y_dev.reset_index(drop=True)


def build_reference_profile(
    reference_data: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    predictor: CreditRiskPredictor | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    settings = monitoring_settings(config)
    predictor = predictor or CreditRiskPredictor()
    features = list(predictor.required_features)

    if reference_data is None:
        reference_data, _labels = recreate_development_population(config)
    _validate_required_columns(reference_data, features)
    reference_features = reference_data.loc[:, features].copy()
    predictions = predictor.predict_batch(reference_features)

    feature_profiles = {}
    numeric_features = _raw_numeric_features(predictor.feature_schema)
    for feature in features:
        if feature in numeric_features:
            feature_profiles[feature] = build_numeric_reference(
                reference_features[feature],
                int(settings["numeric_bins"]),
            )
        else:
            feature_profiles[feature] = build_categorical_reference(reference_features[feature])

    outputs = {
        "pd": build_numeric_reference(predictions["pd"], int(settings["numeric_bins"])),
        "credit_score": build_numeric_reference(predictions["credit_score"], int(settings["numeric_bins"])),
        "risk_grade": build_categorical_reference_with_categories(
            predictions["risk_grade"],
            list(predictor.credit_policy["risk_grade_labels"]),
        ),
        "decision": build_categorical_reference_with_categories(predictions["decision"], DECISION_CATEGORIES),
    }

    profile = {
        "profile_type": "credit_risk_monitoring_reference",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_population": {
            "name": "Development population",
            "definition": "Frozen 80/20 train_test_split development side using config random_state, test_size, and stratification.",
            "uses_locked_test": False,
        },
        "reference_rows": int(len(reference_features)),
        "model_version": predictor.model_metadata.get("artifact_version"),
        "schema_version": predictor.feature_schema.get("schema_version"),
        "policy_status": predictor.credit_policy.get("policy_status"),
        "calibration_method": predictor.credit_policy.get("calibration_method"),
        "feature_list": features,
        "feature_count": int(len(features)),
        "monitoring_config": {
            "numeric_bins": int(settings["numeric_bins"]),
            "epsilon": float(settings["epsilon"]),
            "psi_warning_threshold": float(settings["psi_warning_threshold"]),
            "psi_alert_threshold": float(settings["psi_alert_threshold"]),
            "min_labeled_rows": int(settings["min_labeled_rows"]),
            "threshold_note": "Configurable monitoring heuristic, not a universal regulatory standard.",
        },
        "features": feature_profiles,
        "outputs": outputs,
    }
    if output_path is not None:
        save_json(profile, output_path)
    return profile


def build_categorical_reference_with_categories(
    series: pd.Series,
    categories: list[str],
) -> dict[str, Any]:
    from src.monitoring.drift import categorical_distribution, missing_rate

    return {
        "feature_type": "categorical",
        "categories": categories,
        "proportions": categorical_distribution(series, categories),
        "missing_rate": missing_rate(series),
    }


def load_reference_profile(path: str | Path | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    profile_path = Path(path) if path else reference_profile_path(config)
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Monitoring reference profile not found: {profile_path}. "
            "Run `python -m src.monitoring.monitor build-reference` first."
        )
    with profile_path.open(encoding="utf-8") as handle:
        return _decode_json_values(json.load(handle))


def analyze_current_batch(
    current_data: pd.DataFrame,
    labels: list[int] | pd.Series | None = None,
    reference_profile: dict[str, Any] | None = None,
    reference_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    predictor: CreditRiskPredictor | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    settings = monitoring_settings(config)
    predictor = predictor or CreditRiskPredictor()
    reference_profile = reference_profile or load_reference_profile(reference_path, config)
    _validate_reference_compatibility(reference_profile, predictor)

    features = list(predictor.required_features)
    _validate_required_columns(current_data, features)
    if len(current_data) == 0:
        raise ValueError("current batch must contain at least one applicant row.")
    if len(current_data) > int(settings["max_batch_size"]):
        raise ValueError(f"current batch exceeds max monitoring batch size {settings['max_batch_size']}.")

    current_features = current_data.loc[:, features].copy()
    predictions = predictor.predict_batch(current_features)
    epsilon = float(settings["epsilon"])
    warning = float(settings["psi_warning_threshold"])
    alert = float(settings["psi_alert_threshold"])

    feature_drift = sort_drift_rows(
        [
            compare_feature(
                feature,
                reference_profile["features"][feature],
                current_features[feature],
                epsilon,
                warning,
                alert,
            )
            for feature in features
        ]
    )
    pd_drift = _numeric_output_drift("pd", reference_profile, predictions["pd"], epsilon, warning, alert)
    score_drift = _numeric_output_drift(
        "credit_score",
        reference_profile,
        predictions["credit_score"],
        epsilon,
        warning,
        alert,
        public_name="score",
    )
    risk_grade_drift = _categorical_output_drift(
        "risk_grade",
        reference_profile,
        predictions["risk_grade"],
        epsilon,
        warning,
        alert,
    )
    decision_drift = _categorical_output_drift(
        "decision",
        reference_profile,
        predictions["decision"],
        epsilon,
        warning,
        alert,
    )

    components = [
        *(row["status"] for row in feature_drift),
        pd_drift["pd_status"],
        score_drift["score_status"],
        risk_grade_drift["risk_grade_status"],
        decision_drift["decision_status"],
    ]
    report = {
        "report_type": "credit_risk_monitoring_report",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "monitoring_status": overall_status(list(components)),
        "reference_rows": int(reference_profile["reference_rows"]),
        "current_rows": int(len(current_features)),
        "model_version": predictor.model_metadata.get("artifact_version"),
        "schema_version": predictor.feature_schema.get("schema_version"),
        "policy_status": predictor.credit_policy.get("policy_status"),
        "calibration_method": predictor.credit_policy.get("calibration_method"),
        "reference_population": reference_profile["reference_population"],
        "feature_drift": feature_drift,
        "missingness_drift": [
            {
                "feature": row["feature"],
                "reference_missing_rate": row["reference_missing_rate"],
                "current_missing_rate": row["current_missing_rate"],
                "missing_rate_delta": row["missing_rate_delta"],
                "status": row["status"],
            }
            for row in feature_drift
        ],
        "pd_drift": pd_drift,
        "score_drift": score_drift,
        "risk_grade_drift": risk_grade_drift,
        "decision_drift": decision_drift,
        "performance": performance_monitoring(
            labels,
            predictions["pd"],
            predictor.model_metadata.get("metrics", {}).get("locked_test", {}),
            int(settings["min_labeled_rows"]),
        ),
        "alerts": build_alerts(feature_drift, pd_drift, score_drift, risk_grade_drift, decision_drift),
        "limitations": monitoring_limitations(),
    }
    return _encode_json_values(report)


def analyze_applicants(
    applicants: list[dict[str, Any]],
    labels: list[int] | None = None,
    predictor: CreditRiskPredictor | None = None,
    reference_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = pd.DataFrame(applicants)
    return analyze_current_batch(
        frame,
        labels=labels,
        predictor=predictor,
        reference_path=reference_path,
        config=config,
    )


def performance_monitoring(
    labels: list[int] | pd.Series | None,
    pd_values: pd.Series,
    baseline_metrics: dict[str, Any],
    min_labeled_rows: int,
) -> dict[str, Any]:
    if labels is None:
        return {"performance_status": "labels_unavailable"}
    if len(labels) != len(pd_values):
        return {
            "performance_status": "invalid_labels",
            "reason": "labels length does not match current batch rows.",
        }
    if len(labels) < min_labeled_rows:
        return {
            "performance_status": "insufficient_labels",
            "min_labeled_rows": int(min_labeled_rows),
            "labeled_rows": int(len(labels)),
        }
    try:
        metrics = evaluate_binary_probabilities(labels, pd_values.to_numpy(dtype=float))
    except ValueError as exc:
        return {"performance_status": "invalid_labels", "reason": str(exc)}

    observed_default_rate = float(pd.Series(labels, dtype=int).mean())
    mean_predicted_pd = float(pd_values.mean())
    calibration_gap = float(mean_predicted_pd - observed_default_rate)
    result: dict[str, Any] = {
        "performance_status": "available",
        "labeled_rows": int(len(labels)),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "ks": metrics["ks"],
        "gini": metrics["gini"],
        "brier": metrics["brier"],
        "observed_default_rate": observed_default_rate,
        "mean_predicted_pd": mean_predicted_pd,
        "calibration_gap": calibration_gap,
        "absolute_calibration_gap": abs(calibration_gap),
        "frozen_evaluation_reference": baseline_metrics,
    }
    for metric in ["roc_auc", "pr_auc", "ks", "gini", "brier"]:
        baseline = baseline_metrics.get(metric)
        if baseline is not None:
            result[f"{metric}_delta_vs_frozen_evaluation_reference"] = float(metrics[metric] - baseline)
    return result


def build_alerts(
    feature_drift: list[dict[str, Any]],
    pd_drift: dict[str, Any],
    score_drift: dict[str, Any],
    risk_grade_drift: dict[str, Any],
    decision_drift: dict[str, Any],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in feature_drift:
        if row["status"] in {"WARNING", "ALERT"}:
            alerts.append(
                {
                    "type": "FEATURE_DRIFT",
                    "feature": row["feature"],
                    "severity": row["status"],
                    "psi": row["psi"],
                    "message": "Feature distribution shifted versus the monitoring reference.",
                }
            )
    output_specs = [
        ("PD_DRIFT", None, pd_drift, "pd_psi", "pd_status"),
        ("SCORE_DRIFT", None, score_drift, "score_psi", "score_status"),
        ("RISK_GRADE_DRIFT", None, risk_grade_drift, "risk_grade_psi", "risk_grade_status"),
        ("DECISION_DRIFT", None, decision_drift, "decision_psi", "decision_status"),
    ]
    for alert_type, _feature, payload, psi_key, status_key in output_specs:
        if payload[status_key] in {"WARNING", "ALERT"}:
            alerts.append(
                {
                    "type": alert_type,
                    "severity": payload[status_key],
                    "psi": payload[psi_key],
                    "message": "Model output distribution shifted versus the monitoring reference.",
                }
            )
    return alerts


def monitoring_limitations() -> list[str]:
    return [
        "Reference is derived from the project Development population, not the Locked Test split.",
        "Current batch monitoring is a simulation; the dataset does not provide a real production timestamp stream.",
        "PSI thresholds are configurable heuristics, not universal regulatory standards.",
        "Drift indicates distribution shift and does not establish causal model degradation.",
        "Performance metrics require realized loan_status labels.",
        "Retraining decisions require separate validation and governance.",
    ]


def save_monitoring_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_dir) / f"monitoring_report_{timestamp}.json"
    save_json(report, path)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_encode_json_values(payload), handle, indent=2, ensure_ascii=True, allow_nan=False)
    return path


def reference_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_status": "available",
        "model_version": profile.get("model_version"),
        "schema_version": profile.get("schema_version"),
        "policy_status": profile.get("policy_status"),
        "calibration_method": profile.get("calibration_method"),
        "reference_rows": profile.get("reference_rows"),
        "created_at_utc": profile.get("created_at_utc"),
        "feature_count": profile.get("feature_count"),
        "reference_population": profile.get("reference_population"),
    }


def _numeric_output_drift(
    output_name: str,
    profile: dict[str, Any],
    current: pd.Series,
    epsilon: float,
    warning: float,
    alert: float,
    public_name: str | None = None,
) -> dict[str, Any]:
    name = public_name or output_name
    result = compare_numeric(output_name, profile["outputs"][output_name], current, epsilon, warning, alert)
    return {
        f"{name}_psi": result["psi"],
        f"{name}_status": result["status"],
        f"reference_mean_{name}": result["reference_summary"]["mean"],
        f"current_mean_{name}": result["current_summary"]["mean"],
        f"reference_median_{name}": result["reference_summary"]["median"],
        f"current_median_{name}": result["current_summary"]["median"],
        "reference_distribution": result["reference_distribution"],
        "current_distribution": result["current_distribution"],
        "bin_edges": result["bin_edges"],
    }


def _categorical_output_drift(
    output_name: str,
    profile: dict[str, Any],
    current: pd.Series,
    epsilon: float,
    warning: float,
    alert: float,
) -> dict[str, Any]:
    result = compare_categorical(output_name, profile["outputs"][output_name], current, epsilon, warning, alert)
    payload = {
        f"{output_name}_psi": result["psi"],
        f"{output_name}_status": result["status"],
        "reference_distribution": result["reference_distribution"],
        "current_distribution": result["current_distribution"],
    }
    if output_name == "decision":
        payload.update(
            {
                "reference_approval_rate": result["reference_distribution"].get("APPROVE", 0.0),
                "current_approval_rate": result["current_distribution"].get("APPROVE", 0.0),
                "reference_review_rate": result["reference_distribution"].get("MANUAL_REVIEW", 0.0),
                "current_review_rate": result["current_distribution"].get("MANUAL_REVIEW", 0.0),
                "reference_rejection_rate": result["reference_distribution"].get("REJECT", 0.0),
                "current_rejection_rate": result["current_distribution"].get("REJECT", 0.0),
            }
        )
    return payload


def _raw_numeric_features(feature_schema: dict[str, Any]) -> set[str]:
    engineered_numeric = set(feature_schema.get("numeric_features_after_engineering", []))
    return set(feature_schema["raw_primary_features"]) & engineered_numeric


def _validate_required_columns(frame: pd.DataFrame, features: list[str]) -> None:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError("Missing required applicant fields: " + ", ".join(missing))


def _validate_reference_compatibility(profile: dict[str, Any], predictor: CreditRiskPredictor) -> None:
    mismatches = []
    if profile.get("model_version") != predictor.model_metadata.get("artifact_version"):
        mismatches.append("model_version")
    if profile.get("schema_version") != predictor.feature_schema.get("schema_version"):
        mismatches.append("schema_version")
    if profile.get("policy_status") != predictor.credit_policy.get("policy_status"):
        mismatches.append("policy_status")
    if profile.get("calibration_method") != predictor.credit_policy.get("calibration_method"):
        mismatches.append("calibration_method")
    if mismatches:
        raise ValueError("Monitoring reference is incompatible: " + ", ".join(mismatches))


def _encode_json_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _encode_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_json_values(item) for item in value]
    if isinstance(value, float):
        if value == float("inf"):
            return "__POS_INF__"
        if value == float("-inf"):
            return "__NEG_INF__"
    return value


def _decode_json_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _decode_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_values(item) for item in value]
    if value == "__POS_INF__":
        return float("inf")
    if value == "__NEG_INF__":
        return float("-inf")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Credit-risk monitoring CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-reference", help="Build the monitoring reference profile.")
    build.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    build.add_argument("--output-dir", default=None)

    analyze = subparsers.add_parser("analyze", help="Analyze a current applicant CSV.")
    analyze.add_argument("--input", required=True, help="Path to current batch CSV.")
    analyze.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    analyze.add_argument("--reference", default=None, help="Path to reference_profile.json.")
    analyze.add_argument("--output-dir", default=None)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = monitoring_settings(config)
    output_dir = Path(args.output_dir or settings["output_dir"])

    if args.command == "build-reference":
        path = reference_profile_path(config, output_dir)
        build_reference_profile(config=config, output_path=path)
        print(f"Monitoring reference profile saved: {path}")
        return 0

    current = pd.read_csv(args.input)
    target_col = config["data"]["target_col"]
    labels = current[target_col].tolist() if target_col in current.columns else None
    report = analyze_current_batch(
        current,
        labels=labels,
        reference_path=args.reference,
        config=config,
    )
    path = save_monitoring_report(report, output_dir)
    print(f"Monitoring report saved: {path}")
    print(f"Monitoring status: {report['monitoring_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
