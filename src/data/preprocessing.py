"""Reusable preprocessing components for the frozen credit-risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import add_credit_risk_features


@dataclass(frozen=True)
class FeatureTypeSummary:
    """Column groups after business cleaning and feature engineering."""

    numeric_features: list[str]
    categorical_features: list[str]
    engineered_features: list[str]


class BusinessRuleCleaner(BaseEstimator, TransformerMixin):
    """Stateless business-rule cleaner for use inside sklearn pipelines."""

    def __init__(
        self,
        min_age: float = 20,
        max_age: float = 100,
        minimum_working_age: float = 14,
        original_missing_indicator_cols: tuple[str, ...] = ("person_emp_length",),
    ):
        self.min_age = min_age
        self.max_age = max_age
        self.minimum_working_age = minimum_working_age
        self.original_missing_indicator_cols = original_missing_indicator_cols

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BusinessRuleCleaner requires a pandas DataFrame.")
        self.input_features_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BusinessRuleCleaner requires a pandas DataFrame.")
        if not hasattr(self, "input_features_"):
            raise RuntimeError("BusinessRuleCleaner must be fitted before transform.")

        missing_cols = [col for col in self.input_features_ if col not in X.columns]
        if missing_cols:
            raise KeyError("Missing input columns: " + ", ".join(missing_cols))

        result = X.loc[:, self.input_features_].copy()

        for col in self.original_missing_indicator_cols:
            if col in result.columns:
                result[f"{col}_missing_flag"] = result[col].isna().astype("int8")

        if "person_age" in result.columns:
            age = pd.to_numeric(result["person_age"], errors="coerce")
            invalid_age = ~age.between(self.min_age, self.max_age)
            result.loc[invalid_age, "person_age"] = np.nan

        if "person_emp_length" in result.columns:
            emp_length = pd.to_numeric(result["person_emp_length"], errors="coerce")
            invalid_emp = emp_length < 0
            if "person_age" in result.columns:
                age = pd.to_numeric(result["person_age"], errors="coerce")
                invalid_emp = invalid_emp | (
                    age.notna()
                    & emp_length.notna()
                    & (emp_length > age - self.minimum_working_age)
                )
            result.loc[invalid_emp, "person_emp_length"] = np.nan

        return result


class CreditRiskFeatureBuilder(BaseEstimator, TransformerMixin):
    """Apply approved credit-risk feature engineering inside a pipeline."""

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditRiskFeatureBuilder requires a pandas DataFrame.")
        self.input_features_ = X.columns.tolist()
        self.output_features_ = add_credit_risk_features(X).columns.tolist()
        self.engineered_features_ = [
            col for col in self.output_features_ if col not in self.input_features_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditRiskFeatureBuilder requires a pandas DataFrame.")
        if not hasattr(self, "input_features_"):
            raise RuntimeError("CreditRiskFeatureBuilder must be fitted before transform.")

        missing_cols = [col for col in self.input_features_ if col not in X.columns]
        if missing_cols:
            raise KeyError("Missing input columns: " + ", ".join(missing_cols))

        result = add_credit_risk_features(X.loc[:, self.input_features_])
        return result.reindex(columns=self.output_features_)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create an sklearn-version-compatible dense one-hot encoder."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_types(
    X: pd.DataFrame,
    raw_feature_cols: list[str],
) -> FeatureTypeSummary:
    """Infer model-ready numeric/categorical groups after transformations."""

    cleaned = BusinessRuleCleaner().fit_transform(X.loc[:, raw_feature_cols])
    engineered = CreditRiskFeatureBuilder().fit_transform(cleaned)

    categorical_features = [
        col for col in engineered.columns
        if pd.api.types.is_object_dtype(engineered[col])
        or pd.api.types.is_string_dtype(engineered[col])
        or isinstance(engineered[col].dtype, pd.CategoricalDtype)
    ]
    numeric_features = [
        col for col in engineered.columns if col not in categorical_features
    ]
    engineered_features = [
        col for col in engineered.columns if col not in raw_feature_cols
    ]

    return FeatureTypeSummary(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        engineered_features=engineered_features,
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    use_scaler: bool = False,
) -> ColumnTransformer:
    """Build the preprocessing block used after feature engineering."""

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
