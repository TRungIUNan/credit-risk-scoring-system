from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    BusinessRuleCleaner,
    CreditRiskFeatureBuilder,
    build_preprocessor,
    infer_feature_types,
)


def test_age_outside_frozen_range_is_marked_missing():
    df = pd.DataFrame({"person_age": [19, 20, 100, 101, 144]})

    result = BusinessRuleCleaner().fit_transform(df)

    assert pd.isna(result.loc[0, "person_age"])
    assert result.loc[1, "person_age"] == 20
    assert result.loc[2, "person_age"] == 100
    assert pd.isna(result.loc[3, "person_age"])
    assert pd.isna(result.loc[4, "person_age"])


def test_employment_length_rule_marks_impossible_values_missing():
    df = pd.DataFrame(
        {
            "person_age": [30, 30, 30],
            "person_emp_length": [10, 20, -1],
        }
    )

    result = BusinessRuleCleaner().fit_transform(df)

    assert result.loc[0, "person_emp_length"] == 10
    assert pd.isna(result.loc[1, "person_emp_length"])
    assert pd.isna(result.loc[2, "person_emp_length"])


def test_original_missing_employment_creates_indicator_before_invalid_cleanup():
    df = pd.DataFrame(
        {
            "person_age": [30, 30, 30],
            "person_emp_length": [np.nan, 5, 20],
        }
    )

    result = BusinessRuleCleaner().fit_transform(df)

    assert result["person_emp_length_missing_flag"].tolist() == [1, 0, 0]
    assert pd.isna(result.loc[2, "person_emp_length"])


def test_numeric_imputation_uses_fit_data_median(small_valid_dataframe, config):
    X = small_valid_dataframe.copy()
    X.loc[0, "person_income"] = np.nan
    type_summary = infer_feature_types(X, config["features"]["primary"])
    transformed_features = CreditRiskFeatureBuilder().fit_transform(
        BusinessRuleCleaner().fit_transform(X)
    )
    preprocessor = build_preprocessor(
        type_summary.numeric_features,
        type_summary.categorical_features,
    )

    transformed = preprocessor.fit_transform(transformed_features)

    assert not np.isnan(transformed[:, : len(type_summary.numeric_features)]).any()
    numeric_pipe = preprocessor.named_transformers_["numeric"]
    income_index = type_summary.numeric_features.index("person_income")
    assert numeric_pipe.named_steps["imputer"].statistics_[income_index] == 80000


def test_categorical_imputation_and_unknown_category_do_not_crash(small_valid_dataframe, config):
    X_train = small_valid_dataframe.copy()
    X_train.loc[1, "loan_intent"] = np.nan
    type_summary = infer_feature_types(X_train, config["features"]["primary"])
    train_features = CreditRiskFeatureBuilder().fit_transform(
        BusinessRuleCleaner().fit_transform(X_train)
    )
    preprocessor = build_preprocessor(
        type_summary.numeric_features,
        type_summary.categorical_features,
    ).fit(train_features)

    X_new = small_valid_dataframe.copy()
    X_new.loc[0, "loan_intent"] = "UNKNOWN_NEW_INTENT"
    new_features = CreditRiskFeatureBuilder().fit_transform(
        BusinessRuleCleaner().fit_transform(X_new)
    )

    transformed = preprocessor.transform(new_features)

    assert transformed.shape[0] == len(X_new)


def test_preprocessing_does_not_require_target_or_emit_target(small_valid_dataframe, config):
    type_summary = infer_feature_types(small_valid_dataframe, config["features"]["primary"])

    assert "loan_status" not in type_summary.numeric_features
    assert "loan_status" not in type_summary.categorical_features
