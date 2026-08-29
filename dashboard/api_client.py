"""HTTP client for the Credit Risk FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("CREDIT_RISK_API_URL", DEFAULT_API_BASE_URL)


class CreditRiskAPIError(RuntimeError):
    """Raised when the API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CreditRiskAPIUnavailable(CreditRiskAPIError):
    """Raised when the dashboard cannot reach the API."""


class CreditRiskAPIClient:
    """Small, reusable client for all dashboard-to-API communication."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ):
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model/info")

    def openapi(self) -> dict[str, Any]:
        return self._request("GET", "/openapi.json")

    def predict(self, applicant: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=applicant)

    def predict_batch(self, applicants: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/predict/batch", json={"applicants": applicants})

    def monitoring_reference(self) -> dict[str, Any]:
        return self._request("GET", "/monitoring/reference")

    def monitoring_analyze(
        self,
        applicants: list[dict[str, Any]],
        labels: list[int] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/monitoring/analyze",
            json={"applicants": applicants, "labels": labels},
        )

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                json=json,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise CreditRiskAPIUnavailable(
                f"Credit Risk API request timed out: {url}",
            ) from exc
        except requests.ConnectionError as exc:
            raise CreditRiskAPIUnavailable(
                f"Credit Risk API is currently unavailable: {self.base_url}",
            ) from exc
        except requests.RequestException as exc:
            raise CreditRiskAPIUnavailable(
                f"Credit Risk API request failed: {exc}",
            ) from exc

        if response.status_code >= 400:
            raise CreditRiskAPIError(
                self._extract_error_message(response),
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CreditRiskAPIError(
                f"Credit Risk API returned non-JSON response from {path}.",
                status_code=response.status_code,
            ) from exc

        if not isinstance(payload, dict):
            raise CreditRiskAPIError(
                f"Credit Risk API returned unexpected payload from {path}.",
                status_code=response.status_code,
            )
        return payload

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"Credit Risk API returned HTTP {response.status_code}."

        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            messages = []
            for item in detail:
                if isinstance(item, dict):
                    location = ".".join(str(part) for part in item.get("loc", []))
                    message = item.get("msg", "Invalid request")
                    messages.append(f"{location}: {message}" if location else message)
            if messages:
                return "; ".join(messages)
        return f"Credit Risk API returned HTTP {response.status_code}."
