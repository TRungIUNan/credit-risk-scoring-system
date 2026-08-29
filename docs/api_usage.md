# API Usage

The FastAPI service exposes the frozen credit-risk inference and monitoring layers.

Start from the project root:

```bash
python -m uvicorn api.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check API and model-load status. |
| `GET` | `/model/info` | Return public model metadata. |
| `POST` | `/predict` | Score one applicant. |
| `POST` | `/predict/batch` | Score a batch of applicants. |
| `GET` | `/monitoring/reference` | Return monitoring reference metadata. |
| `POST` | `/monitoring/analyze` | Analyze current batch drift and optional performance. |

There is no `/explain` endpoint in the current API.

## Applicant Request Fields

`ApplicantRequest` accepts exactly the frozen Primary Feature contract:

```json
{
  "person_age": 35,
  "person_income": 65000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
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
  "debt_to_income_ratio": 0.5,
  "open_accounts": 6,
  "credit_utilization_ratio": 0.85,
  "past_delinquencies": 1
}
```

Forbidden fields include `loan_status`, `client_ID`, `loan_grade`, `loan_int_rate`, audit-only variables, and redundant excluded variables.

## `GET /health`

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2026-08-29-frozen-notebook-04",
  "calibration_method": "isotonic",
  "policy_status": "frozen"
}
```

## `GET /model/info`

Returns public metadata such as model family, model version, calibration method, feature count, score range, risk grades, policy status, and locked-test metrics.

## `POST /predict`

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 35,
    "person_income": 65000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
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
    "debt_to_income_ratio": 0.5,
    "open_accounts": 6,
    "credit_utilization_ratio": 0.85,
    "past_delinquencies": 1
  }'
```

Response schema:

```json
{
  "pd": 0.0,
  "credit_score": 0.0,
  "risk_grade": "A",
  "decision": "APPROVE",
  "risk_drivers": null,
  "explanation_status": "explanation_unavailable",
  "model_version": "2026-08-29-frozen-notebook-04",
  "calibration_method": "isotonic",
  "policy_status": "frozen"
}
```

The numeric values above are schema placeholders, not a documented prediction for a real applicant.

## `POST /predict/batch`

Request body:

```json
{
  "applicants": [
    {
      "person_age": 35,
      "person_income": 65000,
      "person_home_ownership": "RENT",
      "person_emp_length": 5,
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
      "debt_to_income_ratio": 0.5,
      "open_accounts": 6,
      "credit_utilization_ratio": 0.85,
      "past_delinquencies": 1
    }
  ]
}
```

Maximum batch size is 1,000 applicants.

## `GET /monitoring/reference`

Returns metadata only:

```json
{
  "reference_status": "available",
  "model_version": "2026-08-29-frozen-notebook-04",
  "schema_version": "2026-08-29-frozen-notebook-04",
  "policy_status": "frozen",
  "calibration_method": "isotonic",
  "reference_rows": 26064,
  "feature_count": 18
}
```

## `POST /monitoring/analyze`

Request body:

```json
{
  "applicants": [
    {
      "person_age": 35,
      "person_income": 65000,
      "person_home_ownership": "RENT",
      "person_emp_length": 5,
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
      "debt_to_income_ratio": 0.5,
      "open_accounts": 6,
      "credit_utilization_ratio": 0.85,
      "past_delinquencies": 1
    }
  ],
  "labels": null
}
```

If realized outcomes are available, pass labels separately:

```json
{
  "applicants": [ "... applicant objects ..." ],
  "labels": [0, 1, 0]
}
```

Labels are never part of `ApplicantRequest` and are separated before inference.

Response contains:

- `monitoring_status`
- `feature_drift`
- `missingness_drift`
- `pd_drift`
- `score_drift`
- `risk_grade_drift`
- `decision_drift`
- `performance`
- `alerts`
- `limitations`

## Common Status Codes

| Status | Meaning |
| --- | --- |
| `200` | Request succeeded. |
| `422` | Invalid request, missing required field, forbidden field, label length mismatch, empty batch, or too-large batch. |
| `503` | Model artifacts or monitoring reference profile unavailable. |
| `500` | Unexpected internal server error. |

## Dashboard Client

The Streamlit dashboard uses `dashboard/api_client.py` and communicates only via HTTP. It does not import predictor, model artifacts, or monitoring formulas directly.
