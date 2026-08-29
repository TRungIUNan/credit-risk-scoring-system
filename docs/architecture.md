# Architecture

This project separates research, reusable ML logic, model artifacts, service APIs, dashboard presentation, and monitoring. The goal is a readable portfolio implementation rather than a real production bank platform.

## Research to Engineering Flow

```mermaid
flowchart LR
    A[Raw Excel Dataset] --> B[01 Data Audit]
    B --> C[02 EDA]
    C --> D[03 Feature Analysis]
    D --> E[04 Model Analysis]
    E --> F[src Reusable Training Pipeline]
    F --> G[Frozen Artifacts in models/]
    G --> H[CreditRiskPredictor]
```

Responsibilities:

| Layer | Responsibility |
| --- | --- |
| `notebooks/` | Research narrative, audits, modeling experiments, calibration, policy design, locked-test evaluation. |
| `src/data/` | Data validation and reusable preprocessing components. |
| `src/features/` | Feature eligibility rules and engineered feature builders. |
| `src/models/` | Training, calibration, evaluation, and score mapping helpers. |
| `src/decision/` | Frozen risk-grade and decision policy utilities. |
| `src/inference/` | Artifact loading, schema validation, and prediction orchestration. |
| `src/monitoring/` | Drift and monitoring report logic. |
| `models/` | Frozen model, schema, policy, and metadata artifacts. |

## Serving Architecture

```mermaid
flowchart LR
    A[User] --> B[Streamlit Dashboard]
    B --> C[FastAPI]
    C --> D[CreditRiskPredictor]
    D --> E[calibrated_model.joblib]
    D --> F[feature_schema.json]
    D --> G[credit_policy.json]
    D --> H[model_metadata.json]
    D --> I[PD / Score / Grade / Decision]
    I --> C
    C --> B
```

Serving boundaries:

- Streamlit is presentation only and calls FastAPI over HTTP.
- FastAPI validates request structure and delegates scoring to `CreditRiskPredictor`.
- `CreditRiskPredictor` loads frozen artifacts once and enforces the frozen feature contract.
- Decision policy is imported from `src/decision/`; thresholds are not duplicated in the API or dashboard.
- Explainability is explicitly unavailable in the serving layer: `risk_drivers = None`.

## Monitoring Architecture

```mermaid
flowchart TD
    A[Development Reference Population] --> B[Reference Profile]
    C[Current Portfolio Batch] --> D[Monitoring Engine]
    B --> D
    D --> E[Feature Drift]
    D --> F[Missingness Drift]
    D --> G[PD Drift]
    D --> H[Score Drift]
    D --> I[Risk Grade Drift]
    D --> J[Decision Drift]
    D --> K[Optional Performance Monitoring]
    E --> L[Monitoring Report]
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M[FastAPI]
    M --> N[Streamlit Monitoring Page]
```

Monitoring boundaries:

- `src/monitoring/drift.py` contains PSI, bucket, missingness, and status calculations.
- `src/monitoring/monitor.py` builds the reference profile, scores current batches through `CreditRiskPredictor`, and produces monitoring reports.
- FastAPI exposes monitoring endpoints but does not calculate PSI directly.
- Dashboard displays monitoring reports and does not import `src.monitoring`.

## Artifact Contract

The frozen inference contract is stored in `models/`:

| Artifact | Purpose |
| --- | --- |
| `champion_model.joblib` | Frozen uncalibrated champion pipeline. |
| `calibrated_model.joblib` | Frozen calibrated model used for prediction. |
| `feature_schema.json` | Raw primary features, engineered features, excluded groups, schema version. |
| `credit_policy.json` | Score config, risk grade thresholds, decision thresholds, policy status. |
| `model_metadata.json` | Data split, model parameters, metrics, artifact metadata. |

The current artifact version is `2026-08-29-frozen-notebook-04`.

## Single Source of Truth

| Concern | Source of truth |
| --- | --- |
| Feature eligibility | `src/features/build_features.py`, `models/feature_schema.json` |
| Preprocessing | `src/data/preprocessing.py` |
| Training and artifact generation | `src/models/train.py` |
| Evaluation metrics | `src/models/evaluate.py` |
| Score mapping | `src/models/scoring.py` |
| Risk grade and decision policy | `src/decision/policy.py`, `models/credit_policy.json` |
| Inference | `src/inference/predictor.py` |
| Monitoring | `src/monitoring/` |
| API schemas and routes | `api/` |
| Dashboard display | `dashboard/` |

## Non-Goals

This architecture does not claim real production readiness. It does not include authentication, authorization, audit logging, a model registry, database persistence, CI/CD, or real-time streaming monitoring.
