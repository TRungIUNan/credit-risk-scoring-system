"""Feature-set decisions and feature builders for credit-risk modeling.

The notebook layer should import these helpers instead of redefining feature
eligibility rules independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COL = "loan_status"
ID_COL = "client_ID"

POTENTIAL_LEAKAGE_COLS = [
    "loan_grade",
    "loan_int_rate",
]

SENSITIVE_OR_PROXY_COLS = [
    "gender",
    "country",
    "state",
    "city",
    "city_latitude",
    "city_longitude",
]


@dataclass(frozen=True)
class FeatureDecisionResult:
    """Container for model feature-set decisions."""

    primary_feature_cols: list[str]
    extended_feature_cols: list[str]
    extended_only_cols: list[str]
    audit_only_cols: list[str]
    redundant_excluded_cols: list[str]
    primary_excluded_cols: pd.DataFrame
    feature_decision_note: str


def build_feature_decisions(
    columns: list[str] | pd.Index,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
    potential_leakage_cols: list[str] | None = None,
    sensitive_or_proxy_cols: list[str] | None = None,
    redundant_excluded_cols: list[str] | None = None,
) -> FeatureDecisionResult:
    """Build primary and extended feature lists from a common rule set."""

    column_list = list(columns)
    leakage_cols = potential_leakage_cols or POTENTIAL_LEAKAGE_COLS
    proxy_cols = sensitive_or_proxy_cols or SENSITIVE_OR_PROXY_COLS
    redundant_cols = redundant_excluded_cols or [
        "loan_percent_income",
        "loan_to_income_ratio_fe",
    ]

    exclusion_reasons = {
        target_col: "Target label, never used as input.",
        id_col: "Identifier, excluded to avoid memorization.",
    }

    for column in leakage_cols:
        exclusion_reasons[column] = (
            "Potential post-underwriting leakage candidate; excluded from primary "
            "model until application-time availability is confirmed."
        )

    for column in proxy_cols:
        exclusion_reasons[column] = (
            "Sensitive/geographic proxy; reserved for fairness/proxy audit and "
            "excluded from primary model by default."
        )

    for column in redundant_cols:
        exclusion_reasons[column] = (
            "Redundant with canonical loan_to_income_ratio in this project; "
            "excluded from primary to avoid carrying duplicate LTI definitions."
        )

    excluded_set = set(exclusion_reasons)

    primary_feature_cols = [
        column for column in column_list
        if column not in excluded_set
    ]

    available_leakage_cols = [
        column for column in leakage_cols
        if column in column_list
    ]

    extended_feature_cols = [
        *primary_feature_cols,
        *[
            column for column in available_leakage_cols
            if column not in primary_feature_cols
        ],
    ]

    extended_only_cols = [
        column for column in extended_feature_cols
        if column not in primary_feature_cols
    ]

    audit_only_cols = [
        column for column in proxy_cols
        if column in column_list
    ]

    available_redundant_cols = [
        column for column in redundant_cols
        if column in column_list
    ]

    primary_excluded_cols = pd.DataFrame([
        {
            "column": column,
            "exclude_reason": reason,
        }
        for column, reason in exclusion_reasons.items()
        if column in column_list
    ])

    feature_decision_note = (
        "Feature decisions come from src/features/build_features.py. Primary "
        "excludes target, ID, potential leakage candidates, sensitive/geographic "
        "proxy variables, and redundant LTI aliases. Extended equals "
        "Primary plus leakage candidates only for ablation. Audit-only variables "
        "are kept outside modeling feature sets."
    )

    return FeatureDecisionResult(
        primary_feature_cols=primary_feature_cols,
        extended_feature_cols=extended_feature_cols,
        extended_only_cols=extended_only_cols,
        audit_only_cols=audit_only_cols,
        redundant_excluded_cols=available_redundant_cols,
        primary_excluded_cols=primary_excluded_cols,
        feature_decision_note=feature_decision_note,
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and return NaN when denominator is missing or zero."""

    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_credit_risk_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent ratio features used by the modeling notebook."""

    result = df.copy()

    if (
        "loan_to_income_ratio" not in result.columns
        and {"loan_amnt", "person_income"}.issubset(result.columns)
    ):
        result["loan_to_income_ratio_fe"] = _safe_divide(
            result["loan_amnt"],
            result["person_income"],
        )

    if {"other_debt", "person_income"}.issubset(result.columns):
        result["existing_debt_to_income_ratio"] = _safe_divide(
            result["other_debt"],
            result["person_income"],
        )

    if (
        "debt_to_income_ratio" not in result.columns
        and {"other_debt", "loan_amnt", "person_income"}.issubset(result.columns)
    ):
        result["total_debt_to_income_ratio"] = _safe_divide(
            result["other_debt"] + result["loan_amnt"],
            result["person_income"],
        )

    if {"loan_amnt", "loan_term_months"}.issubset(result.columns):
        result["loan_amount_per_month"] = _safe_divide(
            result["loan_amnt"],
            result["loan_term_months"],
        )

    return result


def add_credit_risk_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable binary risk flags from documented heuristics."""

    result = df.copy()

    ratio_source = (
        "loan_to_income_ratio"
        if "loan_to_income_ratio" in result.columns
        else "loan_to_income_ratio_fe"
    )
    if ratio_source in result.columns:
        result["high_loan_to_income_flag"] = (
            result[ratio_source] >= 0.35
        ).astype("int8")

    if "debt_to_income_ratio" in result.columns:
        result["high_debt_to_income_flag"] = (
            result["debt_to_income_ratio"] >= 0.40
        ).astype("int8")

    if "existing_debt_to_income_ratio" in result.columns:
        result["high_existing_debt_to_income_flag"] = (
            result["existing_debt_to_income_ratio"] >= 0.30
        ).astype("int8")

    if "cb_person_default_on_file" in result.columns:
        result["previous_default_flag"] = (
            result["cb_person_default_on_file"].astype("string").eq("Y")
        ).astype("int8")

    if "past_delinquencies" in result.columns:
        result["past_delinquency_flag"] = (
            result["past_delinquencies"] > 0
        ).astype("int8")

    if "cb_person_cred_hist_length" in result.columns:
        result["thin_credit_history_flag"] = (
            result["cb_person_cred_hist_length"] <= 2
        ).astype("int8")

    if "credit_utilization_ratio" in result.columns:
        result["high_credit_utilization_flag"] = (
            result["credit_utilization_ratio"] >= 0.80
        ).astype("int8")

    return result


def add_credit_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all approved feature-engineering steps."""

    result = add_credit_risk_ratio_features(df)
    result = add_credit_risk_flag_features(result)
    result = add_credit_risk_interactions(result)
    return result


def add_credit_risk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small, hypothesis-driven set of interaction features."""

    result = df.copy()

    if {"loan_to_income_ratio", "previous_default_flag"}.issubset(result.columns):
        result["loan_to_income_x_previous_default"] = (
            result["loan_to_income_ratio"]
            * result["previous_default_flag"]
        )

    if {"debt_to_income_ratio", "past_delinquency_flag"}.issubset(result.columns):
        result["debt_to_income_x_past_delinquency"] = (
            result["debt_to_income_ratio"]
            * result["past_delinquency_flag"]
        )

    if {"cb_person_cred_hist_length", "credit_utilization_ratio"}.issubset(result.columns):
        result["credit_history_x_credit_utilization"] = (
            result["cb_person_cred_hist_length"]
            * result["credit_utilization_ratio"]
        )

    return result
