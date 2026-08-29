"""Frozen credit policy utilities for model outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CreditPolicy:
    approve_threshold: float = 0.20789473684210527
    reject_threshold: float = 0.71875
    risk_grade_labels: tuple[str, ...] = ("A", "B", "C", "D", "E")
    business_utility: dict[str, float] = field(
        default_factory=lambda: {
            "approve_good": 1.0,
            "approve_bad": -5.0,
            "reject_good": -0.2,
            "reject_bad": 0.0,
            "manual_review_good": -0.05,
            "manual_review_bad": -0.05,
        }
    )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["risk_grade_labels"] = list(self.risk_grade_labels)
        return result


def build_risk_grade_thresholds(
    development_pd,
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> list[float]:
    """Build PD grade cutoffs from development-set calibrated probabilities."""

    pd_values = np.asarray(development_pd, dtype=float)
    thresholds = np.quantile(pd_values, quantiles).astype(float)
    for idx in range(1, len(thresholds)):
        if thresholds[idx] <= thresholds[idx - 1]:
            thresholds[idx] = np.nextafter(thresholds[idx - 1], np.inf)
    return [float(value) for value in thresholds]


def assign_risk_grade(
    probability_default,
    grade_thresholds: list[float],
    labels: tuple[str, ...] = ("A", "B", "C", "D", "E"),
) -> pd.Series:
    """Assign ordered risk grades A-E from calibrated PD."""

    bins = [-np.inf, *grade_thresholds, np.inf]
    return pd.cut(
        pd.Series(probability_default),
        bins=bins,
        labels=list(labels),
        include_lowest=True,
        ordered=True,
    ).astype("string")


def assign_credit_decision(
    probability_default,
    approve_threshold: float,
    reject_threshold: float,
) -> np.ndarray:
    """Assign APPROVE / MANUAL_REVIEW / REJECT decisions."""

    pd_values = np.asarray(probability_default, dtype=float)
    decisions = np.full(pd_values.shape, "MANUAL_REVIEW", dtype=object)
    decisions[pd_values < approve_threshold] = "APPROVE"
    decisions[pd_values >= reject_threshold] = "REJECT"
    return decisions


def calculate_decision_metrics(
    y_true,
    decisions,
    business_utility: dict[str, float],
) -> dict[str, float]:
    """Summarize frozen policy outcomes and utility."""

    y_true = np.asarray(y_true).astype(int)
    decisions = np.asarray(decisions, dtype=object)
    total_defaults = max(int((y_true == 1).sum()), 1)
    good = y_true == 0
    bad = y_true == 1

    approve = decisions == "APPROVE"
    manual = decisions == "MANUAL_REVIEW"
    reject = decisions == "REJECT"

    utility = (
        approve & good
    ).sum() * business_utility["approve_good"]
    utility += (approve & bad).sum() * business_utility["approve_bad"]
    utility += (reject & good).sum() * business_utility["reject_good"]
    utility += (reject & bad).sum() * business_utility["reject_bad"]
    utility += (manual & good).sum() * business_utility["manual_review_good"]
    utility += (manual & bad).sum() * business_utility["manual_review_bad"]

    return {
        "approval_rate": float(approve.mean()),
        "manual_review_rate": float(manual.mean()),
        "rejection_rate": float(reject.mean()),
        "captured_default_rate": float(((manual | reject) & bad).sum() / total_defaults),
        "rejected_default_capture_rate": float((reject & bad).sum() / total_defaults),
        "business_utility": float(utility),
        "business_utility_per_applicant": float(utility / len(y_true)),
    }
