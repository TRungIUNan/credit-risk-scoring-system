"""Reusable inference engine for frozen credit-risk artifacts.

Run a lightweight demo from the project root:
    python -m src.inference.predictor
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.decision.policy import assign_credit_decision, assign_risk_grade
from src.models.scoring import CreditScoreConfig, pd_to_score


class CreditRiskPredictor:
    """Load frozen artifacts once and score new credit applicants."""

    REQUIRED_ARTIFACTS = {
        "calibrated_model": "calibrated_model.joblib",
        "feature_schema": "feature_schema.json",
        "credit_policy": "credit_policy.json",
        "model_metadata": "model_metadata.json",
    }

    def __init__(
        self,
        artifact_dir: str | Path = "models",
        allow_extra_fields: bool = True,
        reject_forbidden_fields: bool = True,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.allow_extra_fields = allow_extra_fields
        self.reject_forbidden_fields = reject_forbidden_fields
        self._load_artifacts()
        self._validate_artifact_contract()

    def predict(self, applicant: dict[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Predict calibrated credit risk for one applicant."""

        frame, ignored_extra_fields = self._coerce_single_applicant(applicant)
        result = self._predict_frame(frame)
        row = result.iloc[0].to_dict()
        row["risk_drivers"] = None
        row["explanation_status"] = "explanation_unavailable"
        row["model_version"] = self.model_metadata.get("artifact_version")
        row["calibration_method"] = self.credit_policy["calibration_method"]
        row["policy_status"] = self.credit_policy["policy_status"]
        row["ignored_extra_fields"] = ignored_extra_fields
        return row

    def predict_batch(self, applicants: pd.DataFrame) -> pd.DataFrame:
        """Predict calibrated credit risk for a batch of applicants."""

        if not isinstance(applicants, pd.DataFrame):
            raise TypeError("predict_batch requires a pandas DataFrame.")
        if applicants.empty:
            raise ValueError("predict_batch requires at least one applicant row.")
        if applicants.columns.duplicated().any():
            duplicates = applicants.columns[applicants.columns.duplicated()].tolist()
            raise ValueError("Input contains duplicate columns: " + ", ".join(duplicates))

        aligned, ignored_extra_fields = self._validate_and_align(applicants)
        result = self._predict_frame(aligned)
        result.index = applicants.index
        result["risk_drivers"] = None
        result["explanation_status"] = "explanation_unavailable"
        result["model_version"] = self.model_metadata.get("artifact_version")
        result["calibration_method"] = self.credit_policy["calibration_method"]
        result["policy_status"] = self.credit_policy["policy_status"]
        result["ignored_extra_fields"] = [ignored_extra_fields for _ in range(len(result))]
        return result

    @property
    def required_features(self) -> list[str]:
        return list(self.feature_schema["raw_primary_features"])

    @property
    def forbidden_fields(self) -> set[str]:
        return {
            self.feature_schema["target_col"],
            self.feature_schema["id_col"],
            *self.feature_schema.get("extended_only_features", []),
            *self.feature_schema.get("audit_only_features", []),
            *self.feature_schema.get("redundant_excluded_features", []),
        }

    def _load_artifacts(self) -> None:
        paths = {
            name: self.artifact_dir / filename
            for name, filename in self.REQUIRED_ARTIFACTS.items()
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required inference artifacts: " + ", ".join(missing))

        self.calibrated_model = joblib.load(paths["calibrated_model"])
        self.feature_schema = self._load_json(paths["feature_schema"])
        self.credit_policy = self._load_json(paths["credit_policy"])
        self.model_metadata = self._load_json(paths["model_metadata"])

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON artifact: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSON artifact must contain an object: {path}")
        return value

    def _validate_artifact_contract(self) -> None:
        schema_required = {
            "schema_version",
            "target_col",
            "id_col",
            "raw_primary_features",
            "extended_only_features",
            "audit_only_features",
        }
        policy_required = {
            "policy_status",
            "calibration_method",
            "approve_threshold",
            "reject_threshold",
            "risk_grade_thresholds",
            "risk_grade_labels",
            "credit_score",
        }
        metadata_required = {"artifact_version", "model", "calibration", "metrics", "artifacts"}

        self._require_keys(self.feature_schema, schema_required, "feature_schema")
        self._require_keys(self.credit_policy, policy_required, "credit_policy")
        self._require_keys(self.model_metadata, metadata_required, "model_metadata")

        if not hasattr(self.calibrated_model, "predict_proba"):
            raise ValueError("calibrated_model artifact must expose predict_proba.")
        if self.credit_policy["policy_status"] != "frozen":
            raise ValueError("credit_policy policy_status must be frozen.")
        if self.credit_policy["calibration_method"] != self.model_metadata["calibration"]["method"]:
            raise ValueError("credit_policy calibration method contradicts model metadata.")
        if self.feature_schema["schema_version"] != self.model_metadata["artifact_version"]:
            raise ValueError("feature_schema version contradicts model metadata.")
        if len(self.required_features) == 0:
            raise ValueError("feature_schema raw_primary_features must not be empty.")
        if len(self.required_features) != len(set(self.required_features)):
            raise ValueError("feature_schema raw_primary_features contains duplicates.")
        if not self.credit_policy["approve_threshold"] < self.credit_policy["reject_threshold"]:
            raise ValueError("approve_threshold must be lower than reject_threshold.")

        score_config = CreditScoreConfig(**self.credit_policy["credit_score"])
        if score_config.score_min >= score_config.score_max:
            raise ValueError("credit_score score_min must be lower than score_max.")

    @staticmethod
    def _require_keys(payload: dict[str, Any], required: set[str], name: str) -> None:
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"{name} artifact is missing required keys: " + ", ".join(missing))

    def _coerce_single_applicant(
        self,
        applicant: dict[str, Any] | pd.Series | pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        if isinstance(applicant, dict):
            frame = pd.DataFrame([applicant.copy()])
        elif isinstance(applicant, pd.Series):
            frame = applicant.to_frame().T
        elif isinstance(applicant, pd.DataFrame):
            if len(applicant) != 1:
                raise ValueError("predict requires exactly one applicant row.")
            frame = applicant.copy(deep=True)
        else:
            raise TypeError("predict requires a dict, pandas Series, or one-row DataFrame.")

        return self._validate_and_align(frame)

    def _validate_and_align(self, applicants: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if applicants.columns.duplicated().any():
            duplicates = applicants.columns[applicants.columns.duplicated()].tolist()
            raise ValueError("Input contains duplicate columns: " + ", ".join(duplicates))

        input_columns = set(applicants.columns)
        required = set(self.required_features)
        missing = sorted(required - input_columns)
        if missing:
            raise ValueError("Missing required applicant fields: " + ", ".join(missing))

        forbidden = sorted(input_columns & self.forbidden_fields)
        if forbidden and self.reject_forbidden_fields:
            raise ValueError("Forbidden applicant fields are not allowed: " + ", ".join(forbidden))

        extra_fields = sorted(input_columns - required - self.forbidden_fields)
        if extra_fields and not self.allow_extra_fields:
            raise ValueError("Unexpected applicant fields: " + ", ".join(extra_fields))

        return applicants.loc[:, self.required_features].copy(), extra_fields

    def _predict_frame(self, aligned_applicants: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.calibrated_model.predict_proba(aligned_applicants)
        if probabilities.ndim != 2 or probabilities.shape != (len(aligned_applicants), 2):
            raise ValueError("calibrated_model returned an invalid probability matrix shape.")

        pd_values = np.asarray(probabilities[:, 1], dtype=float)
        if not np.isfinite(pd_values).all():
            raise ValueError("calibrated_model returned NaN or infinite PD values.")
        if ((pd_values < 0.0) | (pd_values > 1.0)).any():
            raise ValueError("calibrated_model returned PD values outside [0, 1].")

        score_config = CreditScoreConfig(**self.credit_policy["credit_score"])
        scores = pd_to_score(pd_values, score_config)
        grades = assign_risk_grade(
            pd_values,
            self.credit_policy["risk_grade_thresholds"],
            tuple(self.credit_policy["risk_grade_labels"]),
        )
        decisions = assign_credit_decision(
            pd_values,
            self.credit_policy["approve_threshold"],
            self.credit_policy["reject_threshold"],
        )

        return pd.DataFrame(
            {
                "pd": pd_values.astype(float),
                "credit_score": scores.astype(float),
                "risk_grade": grades.astype(str).to_numpy(),
                "decision": decisions.astype(str),
            },
            index=aligned_applicants.index,
        )


def _demo_applicant() -> dict[str, Any]:
    return {
        "person_age": 35,
        "person_income": 65000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 18000,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
        "marital_status": "single",
        "education_level": "Bachelor",
        "employment_type": "full_time",
        "loan_term_months": 36,
        "loan_to_income_ratio": 0.276923,
        "other_debt": 20000,
        "debt_to_income_ratio": 0.50,
        "open_accounts": 6,
        "credit_utilization_ratio": 0.85,
        "past_delinquencies": 1,
    }


def main() -> int:
    predictor = CreditRiskPredictor()
    result = predictor.predict(_demo_applicant())
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
