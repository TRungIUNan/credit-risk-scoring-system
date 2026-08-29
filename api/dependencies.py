"""FastAPI dependencies for the credit-risk API."""

from __future__ import annotations

from fastapi import Request

from src.inference.predictor import CreditRiskPredictor


def get_predictor(request: Request) -> CreditRiskPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise RuntimeError("CreditRiskPredictor is not loaded.")
    return predictor
