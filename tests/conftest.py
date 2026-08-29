from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.validation import EXPECTED_COLUMNS


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config(project_root: Path) -> dict:
    with (project_root / "configs" / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="session")
def feature_schema(project_root: Path) -> dict:
    with (project_root / "models" / "feature_schema.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def credit_policy(project_root: Path) -> dict:
    with (project_root / "models" / "credit_policy.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def model_metadata(project_root: Path) -> dict:
    with (project_root / "models" / "model_metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture()
def small_valid_dataframe(config: dict) -> pd.DataFrame:
    data = {
        "person_age": [25, 40, 60, 35],
        "person_income": [50000, 120000, 80000, 65000],
        "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT"],
        "person_emp_length": [2.0, np.nan, 10.0, 5.0],
        "loan_intent": ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE"],
        "loan_amnt": [10000, 24000, 12000, 18000],
        "cb_person_default_on_file": ["N", "Y", "N", "Y"],
        "cb_person_cred_hist_length": [2, 8, 15, 4],
        "marital_status": ["single", "married", "single", "divorced"],
        "education_level": ["Bachelor", "Master", "High School", "Bachelor"],
        "employment_type": ["full_time", "self_employed", "part_time", "full_time"],
        "loan_term_months": [36, 60, 48, 36],
        "loan_to_income_ratio": [0.20, 0.20, 0.15, 0.276923],
        "other_debt": [5000, 30000, 10000, 20000],
        "debt_to_income_ratio": [0.30, 0.45, 0.25, 0.50],
        "open_accounts": [4, 8, 5, 6],
        "credit_utilization_ratio": [0.40, 0.90, 0.20, 0.85],
        "past_delinquencies": [0, 2, 0, 1],
    }
    return pd.DataFrame(data, columns=config["features"]["primary"])


@pytest.fixture()
def sample_applicants(small_valid_dataframe: pd.DataFrame) -> pd.DataFrame:
    return small_valid_dataframe.head(3).copy()


@pytest.fixture()
def full_schema_rows() -> list[dict]:
    base = {
        "client_ID": "C001",
        "person_age": 30,
        "person_income": 75000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 15000,
        "loan_int_rate": 11.0,
        "loan_status": 0,
        "loan_percent_income": 0.20,
        "cb_person_default_on_file": "N",
        "open_accounts": 5,
        "gender": "female",
        "marital_status": "single",
        "education_level": "Bachelor",
        "country": "US",
        "state": "CA",
        "city": "Los Angeles",
        "city_latitude": 34.05,
        "city_longitude": -118.24,
        "employment_type": "full_time",
        "loan_term_months": 36,
        "credit_utilization_ratio": 0.3,
        "other_debt": 10000,
        "cb_person_cred_hist_length": 4,
        "loan_to_income_ratio": 0.20,
        "debt_to_income_ratio": 0.33,
        "past_delinquencies": 0,
    }
    rows = []
    for idx in range(4):
        row = base.copy()
        row["client_ID"] = f"C{idx + 1:03d}"
        row["loan_status"] = idx % 2
        rows.append(row)
    assert list(rows[0]) == EXPECTED_COLUMNS
    return rows
