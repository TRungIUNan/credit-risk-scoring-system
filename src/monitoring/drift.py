"""Drift calculations for credit-risk monitoring.

PSI thresholds are configurable heuristics for this portfolio simulation;
they are not a universal regulatory standard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MISSING_BUCKET = "__MISSING__"
OTHER_BUCKET = "__OTHER__"
STATUS_ORDER = {"ALERT": 0, "WARNING": 1, "STABLE": 2}


def population_stability_index(
    reference_pct: list[float] | np.ndarray,
    current_pct: list[float] | np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI with epsilon smoothing to avoid log(0)."""

    reference = np.asarray(reference_pct, dtype=float)
    current = np.asarray(current_pct, dtype=float)
    if reference.shape != current.shape:
        raise ValueError("reference_pct and current_pct must have the same shape.")
    reference = np.clip(reference, epsilon, None)
    current = np.clip(current, epsilon, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def psi_status(value: float, warning_threshold: float = 0.10, alert_threshold: float = 0.25) -> str:
    if value >= alert_threshold:
        return "ALERT"
    if value >= warning_threshold:
        return "WARNING"
    return "STABLE"


def missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def numeric_bin_edges(series: pd.Series, n_bins: int = 10) -> list[float]:
    """Create deterministic quantile bin edges from reference data only."""

    values = _clean_numeric(series).dropna()
    if values.empty:
        return [float("-inf"), float("inf")]
    quantiles = np.linspace(0.0, 1.0, max(int(n_bins), 1) + 1)
    inner = np.quantile(values.to_numpy(dtype=float), quantiles)
    unique_inner = sorted({float(edge) for edge in inner if np.isfinite(edge)})
    if len(unique_inner) <= 1:
        return [float("-inf"), float("inf")]
    return [float("-inf"), *unique_inner[1:-1], float("inf")]


def numeric_distribution(series: pd.Series, bin_edges: list[float]) -> dict[str, float]:
    values = _clean_numeric(series)
    labels = [f"bin_{idx:02d}" for idx in range(len(bin_edges) - 1)]
    binned = pd.cut(values, bins=bin_edges, labels=labels, include_lowest=True)
    counts = binned.astype("object").where(values.notna(), MISSING_BUCKET).value_counts()
    total = max(len(values), 1)
    distribution = {label: float(counts.get(label, 0) / total) for label in labels}
    distribution[MISSING_BUCKET] = float(counts.get(MISSING_BUCKET, 0) / total)
    return distribution


def build_numeric_reference(series: pd.Series, n_bins: int = 10) -> dict[str, Any]:
    edges = numeric_bin_edges(series, n_bins)
    values = _clean_numeric(series)
    return {
        "feature_type": "numeric",
        "bin_edges": edges,
        "proportions": numeric_distribution(series, edges),
        "missing_rate": missing_rate(values),
        "summary": {
            "mean": _safe_float(values.mean()),
            "median": _safe_float(values.median()),
        },
    }


def categorical_levels(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str)
    return sorted(values.unique().tolist())


def categorical_distribution(series: pd.Series, categories: list[str]) -> dict[str, float]:
    allowed = set(categories)
    mapped = []
    for value in series.tolist():
        if pd.isna(value):
            mapped.append(MISSING_BUCKET)
        else:
            text = str(value)
            mapped.append(text if text in allowed else OTHER_BUCKET)
    counts = pd.Series(mapped, dtype="object").value_counts()
    total = max(len(series), 1)
    keys = [*categories, OTHER_BUCKET, MISSING_BUCKET]
    return {key: float(counts.get(key, 0) / total) for key in keys}


def build_categorical_reference(series: pd.Series) -> dict[str, Any]:
    categories = categorical_levels(series)
    return {
        "feature_type": "categorical",
        "categories": categories,
        "proportions": categorical_distribution(series, categories),
        "missing_rate": missing_rate(series),
    }


def compare_numeric(
    name: str,
    reference: dict[str, Any],
    current: pd.Series,
    epsilon: float,
    warning_threshold: float,
    alert_threshold: float,
) -> dict[str, Any]:
    current_dist = numeric_distribution(current, reference["bin_edges"])
    current_values = _clean_numeric(current)
    return _comparison_payload(
        name,
        "numeric",
        reference["proportions"],
        current_dist,
        reference["missing_rate"],
        missing_rate(current_values),
        epsilon,
        warning_threshold,
        alert_threshold,
        bin_edges=reference["bin_edges"],
        reference_summary=reference.get("summary"),
        current_summary={
            "mean": _safe_float(current_values.mean()),
            "median": _safe_float(current_values.median()),
        },
    )


def compare_categorical(
    name: str,
    reference: dict[str, Any],
    current: pd.Series,
    epsilon: float,
    warning_threshold: float,
    alert_threshold: float,
) -> dict[str, Any]:
    current_dist = categorical_distribution(current, reference["categories"])
    return _comparison_payload(
        name,
        "categorical",
        reference["proportions"],
        current_dist,
        reference["missing_rate"],
        missing_rate(current),
        epsilon,
        warning_threshold,
        alert_threshold,
        categories=[*reference["categories"], OTHER_BUCKET, MISSING_BUCKET],
    )


def compare_feature(
    name: str,
    reference: dict[str, Any],
    current: pd.Series,
    epsilon: float,
    warning_threshold: float,
    alert_threshold: float,
) -> dict[str, Any]:
    if reference["feature_type"] == "numeric":
        return compare_numeric(name, reference, current, epsilon, warning_threshold, alert_threshold)
    return compare_categorical(name, reference, current, epsilon, warning_threshold, alert_threshold)


def sort_drift_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (STATUS_ORDER.get(row["status"], 3), -float(row["psi"])))


def overall_status(component_statuses: list[str]) -> str:
    if "ALERT" in component_statuses:
        return "ALERT"
    if "WARNING" in component_statuses:
        return "WARNING"
    return "STABLE"


def _comparison_payload(
    name: str,
    feature_type: str,
    reference_dist: dict[str, float],
    current_dist: dict[str, float],
    reference_missing_rate: float,
    current_missing_rate: float,
    epsilon: float,
    warning_threshold: float,
    alert_threshold: float,
    **extra: Any,
) -> dict[str, Any]:
    keys = list(reference_dist)
    reference_values = [float(reference_dist[key]) for key in keys]
    current_values = [float(current_dist.get(key, 0.0)) for key in keys]
    psi = population_stability_index(reference_values, current_values, epsilon)
    payload = {
        "feature": name,
        "feature_type": feature_type,
        "psi": psi,
        "reference_missing_rate": float(reference_missing_rate),
        "current_missing_rate": float(current_missing_rate),
        "missing_rate_delta": float(current_missing_rate - reference_missing_rate),
        "status": psi_status(psi, warning_threshold, alert_threshold),
        "reference_distribution": {key: float(reference_dist[key]) for key in keys},
        "current_distribution": {key: float(current_dist.get(key, 0.0)) for key in keys},
    }
    payload.update(extra)
    return payload


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
