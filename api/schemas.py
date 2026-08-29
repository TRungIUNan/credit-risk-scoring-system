"""Pydantic request/response schemas for the credit-risk API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_BATCH_SIZE = 1000
MAX_MONITORING_BATCH_SIZE = 1000


class ApplicantRequest(BaseModel):
    """Frozen primary feature contract for one applicant."""

    model_config = ConfigDict(extra="forbid")

    person_age: float = Field(description="Applicant age. Business-rule cleaning is handled by the frozen pipeline.")
    person_income: float = Field(description="Applicant annual income.")
    person_home_ownership: str = Field(description="Home ownership category.")
    person_emp_length: float | None = Field(default=None, description="Employment length in years; nullable.")
    loan_intent: str = Field(description="Loan purpose category.")
    loan_amnt: float = Field(description="Requested loan amount.")
    cb_person_default_on_file: str = Field(description="Prior default flag from credit bureau, typically Y/N.")
    cb_person_cred_hist_length: float = Field(description="Credit history length.")
    marital_status: str = Field(description="Marital status category.")
    education_level: str = Field(description="Education level category.")
    employment_type: str = Field(description="Employment type category.")
    loan_term_months: float = Field(description="Loan term in months.")
    loan_to_income_ratio: float = Field(description="Loan amount divided by income.")
    other_debt: float = Field(description="Existing other debt.")
    debt_to_income_ratio: float = Field(description="Debt-to-income ratio from the frozen feature contract.")
    open_accounts: float = Field(description="Number of open accounts.")
    credit_utilization_ratio: float = Field(description="Credit utilization ratio.")
    past_delinquencies: float = Field(description="Count of past delinquencies.")


class PredictionResponse(BaseModel):
    pd: float = Field(description="Calibrated Probability of Default.")
    credit_score: float = Field(description="Internal project credit score; higher means lower modeled risk.")
    risk_grade: str = Field(description="Internal risk grade from A lowest risk to E highest risk.")
    decision: Literal["APPROVE", "MANUAL_REVIEW", "REJECT"] = Field(description="Frozen policy decision.")
    risk_drivers: list[dict[str, Any]] | None = Field(default=None, description="Risk drivers when available.")
    explanation_status: str = Field(description="Explanation availability status.")
    model_version: str | None = Field(default=None, description="Frozen model artifact version.")
    calibration_method: str = Field(description="Probability calibration method.")
    policy_status: str = Field(description="Credit policy status.")


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicants: list[ApplicantRequest] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Applicant records. Maximum batch size is {MAX_BATCH_SIZE}.",
    )


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class MonitoringAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicants: list[ApplicantRequest] = Field(
        min_length=1,
        max_length=MAX_MONITORING_BATCH_SIZE,
        description=f"Current applicant batch. Maximum batch size is {MAX_MONITORING_BATCH_SIZE}.",
    )
    labels: list[int] | None = Field(
        default=None,
        description="Optional realized loan_status labels for performance monitoring.",
    )

    @model_validator(mode="after")
    def labels_match_applicants(self) -> "MonitoringAnalyzeRequest":
        if self.labels is not None and len(self.labels) != len(self.applicants):
            raise ValueError("labels length must match applicants length.")
        return self


class MonitoringReportResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    monitoring_status: Literal["STABLE", "WARNING", "ALERT"]
    reference_rows: int
    current_rows: int
    feature_drift: list[dict[str, Any]]
    pd_drift: dict[str, Any]
    score_drift: dict[str, Any]
    risk_grade_drift: dict[str, Any]
    decision_drift: dict[str, Any]
    performance: dict[str, Any]
    alerts: list[dict[str, Any]]


class MonitoringReferenceResponse(BaseModel):
    reference_status: str
    model_version: str | None = None
    schema_version: str | None = None
    policy_status: str | None = None
    calibration_method: str | None = None
    reference_rows: int | None = None
    created_at_utc: str | None = None
    feature_count: int | None = None
    reference_population: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str | None = None
    calibration_method: str | None = None
    policy_status: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str | None = None
    model_family: str | None = None
    model_version: str | None = None
    calibration_method: str
    raw_feature_count: int
    score_range: dict[str, float | int]
    risk_grades: list[str]
    policy_status: str
    locked_test_metrics: dict[str, float] | None = None
