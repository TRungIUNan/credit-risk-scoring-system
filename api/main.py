"""FastAPI layer for the frozen credit-risk inference engine."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_predictor
from api.schemas import (
    ApplicantRequest,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringAnalyzeRequest,
    MonitoringReferenceResponse,
    MonitoringReportResponse,
    PredictionResponse,
)
from src.inference.predictor import CreditRiskPredictor
from src.monitoring.monitor import analyze_applicants, load_reference_profile, reference_metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = CreditRiskPredictor()
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Credit Risk Scoring API for portfolio/demo project using frozen model artifacts.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    if "CreditRiskPredictor is not loaded" in str(exc):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Credit risk predictor is not available."},
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


def _http_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _prediction_payload(result: dict) -> PredictionResponse:
    public_result = {
        key: result[key]
        for key in PredictionResponse.model_fields
        if key in result
    }
    return PredictionResponse(**public_result)


@app.get("/health", response_model=HealthResponse)
def health(predictor: CreditRiskPredictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.calibrated_model is not None,
        model_version=predictor.model_metadata.get("artifact_version"),
        calibration_method=predictor.credit_policy.get("calibration_method"),
        policy_status=predictor.credit_policy.get("policy_status"),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(predictor: CreditRiskPredictor = Depends(get_predictor)) -> ModelInfoResponse:
    score_config = predictor.credit_policy["credit_score"]
    score_range = {
        "min": score_config["score_min"],
        "max": score_config["score_max"],
    }
    return ModelInfoResponse(
        model_name=predictor.model_metadata.get("model", {}).get("champion_name"),
        model_family=predictor.model_metadata.get("model", {}).get("champion_key"),
        model_version=predictor.model_metadata.get("artifact_version"),
        calibration_method=predictor.credit_policy["calibration_method"],
        raw_feature_count=len(predictor.required_features),
        score_range=score_range,
        risk_grades=list(predictor.credit_policy["risk_grade_labels"]),
        policy_status=predictor.credit_policy["policy_status"],
        locked_test_metrics=predictor.model_metadata.get("metrics", {}).get("locked_test"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: ApplicantRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
) -> PredictionResponse:
    try:
        result = predictor.predict(request.model_dump())
    except ValueError as exc:
        raise _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except FileNotFoundError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "Inference artifact is unavailable.") from exc
    return _prediction_payload(result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    try:
        frame = pd.DataFrame([applicant.model_dump() for applicant in request.applicants])
        result = predictor.predict_batch(frame)
    except ValueError as exc:
        raise _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except FileNotFoundError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "Inference artifact is unavailable.") from exc

    predictions = [
        _prediction_payload(row)
        for row in result.to_dict(orient="records")
    ]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


@app.get("/monitoring/reference", response_model=MonitoringReferenceResponse)
def monitoring_reference(
    predictor: CreditRiskPredictor = Depends(get_predictor),
) -> MonitoringReferenceResponse:
    try:
        profile = load_reference_profile()
    except FileNotFoundError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    try:
        if profile.get("model_version") != predictor.model_metadata.get("artifact_version"):
            raise ValueError("Monitoring reference model_version does not match loaded model.")
    except ValueError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return MonitoringReferenceResponse(**reference_metadata(profile))


@app.post("/monitoring/analyze", response_model=MonitoringReportResponse)
def monitoring_analyze(
    request: MonitoringAnalyzeRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
) -> MonitoringReportResponse:
    try:
        report = analyze_applicants(
            [applicant.model_dump() for applicant in request.applicants],
            labels=request.labels,
            predictor=predictor,
        )
    except ValueError as exc:
        raise _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except FileNotFoundError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return MonitoringReportResponse(**report)
