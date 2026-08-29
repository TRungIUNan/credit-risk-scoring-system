"""Pure helpers for dashboard formatting and portfolio summaries."""

from __future__ import annotations

from math import ceil
from typing import Any, Iterable

import numpy as np
import pandas as pd


DECISION_ORDER = ["APPROVE", "MANUAL_REVIEW", "REJECT"]
DEFAULT_GRADE_ORDER = ["A", "B", "C", "D", "E"]
DRIFT_STATUS_ORDER = {"ALERT": 0, "WARNING": 1, "STABLE": 2}


def format_pd(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def format_score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.0f}"


def normalize_records_for_json(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        row = {}
        for key, value in record.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (np.integer,)):
                row[key] = int(value)
            elif isinstance(value, (np.floating,)):
                row[key] = float(value)
            else:
                row[key] = value
        normalized.append(row)
    return normalized


def chunk_records(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [
        records[start:start + chunk_size]
        for start in range(0, len(records), chunk_size)
    ]


def required_applicant_fields(openapi_schema: dict[str, Any]) -> list[str]:
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    applicant_schema = schemas.get("ApplicantRequest", {})
    fields = applicant_schema.get("properties", {})
    required = applicant_schema.get("required", list(fields))
    return [field for field in required if field in fields]


def batch_limit_from_openapi(openapi_schema: dict[str, Any], default: int = 1000) -> int:
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    batch_schema = schemas.get("BatchPredictionRequest", {})
    applicants = batch_schema.get("properties", {}).get("applicants", {})
    max_items = applicants.get("maxItems")
    return int(max_items) if max_items else default


def validate_portfolio_columns(df: pd.DataFrame, required_fields: list[str]) -> list[str]:
    return [field for field in required_fields if field not in df.columns]


def merge_predictions(inputs: pd.DataFrame, predictions: list[dict[str, Any]]) -> pd.DataFrame:
    if len(inputs) != len(predictions):
        raise ValueError("Prediction count does not match input row count.")
    result = inputs.reset_index(drop=True).copy()
    prediction_frame = pd.DataFrame(predictions).reset_index(drop=True)
    public_cols = ["pd", "credit_score", "risk_grade", "decision"]
    return pd.concat([result, prediction_frame.loc[:, public_cols]], axis=1)


def portfolio_kpis(predictions: pd.DataFrame) -> dict[str, float | int]:
    if predictions.empty:
        raise ValueError("predictions must not be empty.")
    total = len(predictions)
    decisions = predictions["decision"]
    return {
        "total_applicants": int(total),
        "mean_pd": float(predictions["pd"].mean()),
        "median_pd": float(predictions["pd"].median()),
        "average_credit_score": float(predictions["credit_score"].mean()),
        "approval_rate": float((decisions == "APPROVE").mean()),
        "manual_review_rate": float((decisions == "MANUAL_REVIEW").mean()),
        "rejection_rate": float((decisions == "REJECT").mean()),
    }


def ordered_counts(values: pd.Series, order: list[str]) -> pd.Series:
    return values.value_counts().reindex(order, fill_value=0)


def batch_count(total_rows: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return ceil(total_rows / chunk_size) if total_rows else 0


def split_optional_labels(
    df: pd.DataFrame,
    required_fields: list[str],
    target_col: str = "loan_status",
) -> tuple[pd.DataFrame, list[int] | None]:
    labels = df[target_col].astype(int).tolist() if target_col in df.columns else None
    return df.loc[:, required_fields].copy(), labels


def monitoring_summary(report: dict[str, Any]) -> dict[str, Any]:
    feature_rows = report.get("feature_drift", [])
    alert_count = sum(1 for row in feature_rows if row.get("status") == "ALERT")
    warning_count = sum(1 for row in feature_rows if row.get("status") == "WARNING")
    return {
        "monitoring_status": report.get("monitoring_status", "-"),
        "current_rows": report.get("current_rows", 0),
        "alert_feature_count": alert_count,
        "warning_feature_count": warning_count,
        "pd_psi": report.get("pd_drift", {}).get("pd_psi"),
        "score_psi": report.get("score_drift", {}).get("score_psi"),
        "risk_grade_psi": report.get("risk_grade_drift", {}).get("risk_grade_psi"),
        "decision_psi": report.get("decision_drift", {}).get("decision_psi"),
    }


def sorted_feature_drift(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        feature_rows,
        key=lambda row: (DRIFT_STATUS_ORDER.get(row.get("status", "STABLE"), 3), -float(row.get("psi", 0.0))),
    )


def top_feature_psi(feature_rows: list[dict[str, Any]], n: int = 10) -> pd.DataFrame:
    rows = sorted_feature_drift(feature_rows)[:n]
    return pd.DataFrame(
        [
            {
                "feature": row.get("feature"),
                "psi": row.get("psi"),
                "status": row.get("status"),
            }
            for row in rows
        ]
    )


def distribution_comparison_frame(
    drift_payload: dict[str, Any],
    bucket_name: str,
) -> pd.DataFrame:
    reference = drift_payload.get("reference_distribution", {})
    current = drift_payload.get("current_distribution", {})
    buckets = list(reference.keys())
    return pd.DataFrame(
        [
            {
                bucket_name: bucket,
                "reference_pct": reference.get(bucket, 0.0),
                "current_pct": current.get(bucket, 0.0),
            }
            for bucket in buckets
        ]
    )
