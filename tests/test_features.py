from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.build_features import add_credit_risk_features


def test_ratio_features_follow_frozen_formulas():
    df = pd.DataFrame(
        {
            "person_income": [100000],
            "loan_amnt": [20000],
            "loan_term_months": [40],
            "other_debt": [10000],
            "loan_to_income_ratio": [0.20],
            "debt_to_income_ratio": [0.30],
        }
    )

    result = add_credit_risk_features(df)

    assert result.loc[0, "loan_to_income_ratio"] == 0.20
    assert result.loc[0, "existing_debt_to_income_ratio"] == 0.10
    assert result.loc[0, "loan_amount_per_month"] == 500
    assert "total_debt_to_income_ratio" not in result.columns


def test_safe_division_does_not_create_infinite_values():
    df = pd.DataFrame(
        {
            "person_income": [0],
            "loan_amnt": [20000],
            "loan_term_months": [0],
            "other_debt": [10000],
        }
    )

    result = add_credit_risk_features(df)

    numeric = result.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    assert pd.isna(result.loc[0, "loan_to_income_ratio_fe"])
    assert pd.isna(result.loc[0, "loan_amount_per_month"])


def test_risk_flags_follow_boundaries():
    df = pd.DataFrame(
        {
            "loan_to_income_ratio": [0.34, 0.35],
            "debt_to_income_ratio": [0.39, 0.40],
            "other_debt": [29, 30],
            "person_income": [100, 100],
            "cb_person_default_on_file": ["N", "Y"],
            "past_delinquencies": [0, 1],
            "cb_person_cred_hist_length": [3, 2],
            "credit_utilization_ratio": [0.79, 0.80],
        }
    )

    result = add_credit_risk_features(df)

    assert result["high_loan_to_income_flag"].tolist() == [0, 1]
    assert result["high_debt_to_income_flag"].tolist() == [0, 1]
    assert result["high_existing_debt_to_income_flag"].tolist() == [0, 1]
    assert result["previous_default_flag"].tolist() == [0, 1]
    assert result["past_delinquency_flag"].tolist() == [0, 1]
    assert result["thin_credit_history_flag"].tolist() == [0, 1]
    assert result["high_credit_utilization_flag"].tolist() == [0, 1]


def test_interaction_features_have_manual_expected_values():
    df = pd.DataFrame(
        {
            "loan_to_income_ratio": [0.25],
            "debt_to_income_ratio": [0.50],
            "other_debt": [10],
            "person_income": [100],
            "loan_amnt": [20],
            "loan_term_months": [10],
            "cb_person_default_on_file": ["Y"],
            "past_delinquencies": [2],
            "cb_person_cred_hist_length": [6],
            "credit_utilization_ratio": [0.75],
        }
    )

    result = add_credit_risk_features(df)

    assert result.loc[0, "loan_to_income_x_previous_default"] == 0.25
    assert result.loc[0, "debt_to_income_x_past_delinquency"] == 0.50
    assert result.loc[0, "credit_history_x_credit_utilization"] == 4.5


def test_feature_builder_does_not_create_duplicate_columns_or_lti_alias(small_valid_dataframe):
    result = add_credit_risk_features(small_valid_dataframe)

    assert result.columns.is_unique
    assert not {"loan_to_income_ratio", "loan_to_income_ratio_fe"}.issubset(result.columns)


def test_feature_builder_does_not_mutate_input(small_valid_dataframe):
    original = small_valid_dataframe.copy(deep=True)

    add_credit_risk_features(small_valid_dataframe)

    assert_frame_equal(small_valid_dataframe, original)
