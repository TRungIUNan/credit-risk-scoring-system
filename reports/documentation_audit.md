# Documentation Source-of-Truth Audit

This audit records the main claims used in the Task 7 documentation refresh.

| Claim | Source | Documentation files | Status |
| --- | --- | --- | --- |
| Champion model is LightGBM | `models/model_metadata.json`, `configs/config.yaml` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Model artifact version is `2026-08-29-frozen-notebook-04` | `models/model_metadata.json`, `models/feature_schema.json` | `README.md`, `MODEL_CARD.md`, `docs/architecture.md`, `docs/api_usage.md` | PASS |
| Calibration method is isotonic | `models/model_metadata.json`, `models/credit_policy.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md`, `docs/api_usage.md` | PASS |
| Dataset has 32,581 rows and 29 columns | `models/model_metadata.json`, `reports/data_quality_report.md` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Target is `loan_status` | `models/feature_schema.json`, `reports/data_quality_report.md` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md`, `docs/api_usage.md` | PASS |
| Default rate is 21.82% | `models/model_metadata.json`, `reports/data_quality_report.md` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Development rows = 26,064 and Locked Test rows = 6,517 | `models/model_metadata.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md`, `docs/monitoring.md` | PASS |
| Primary raw feature count = 18 | `models/feature_schema.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md`, `docs/api_usage.md` | PASS |
| Engineered feature count = 13 | `models/feature_schema.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| `loan_grade` and `loan_int_rate` are extended-only leakage candidates | `models/feature_schema.json`, `src/features/build_features.py` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Audit-only variables are excluded from model input | `models/feature_schema.json`, `src/features/build_features.py` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Locked Test ROC-AUC = 0.8969 | `models/model_metadata.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Locked Test PR-AUC = 0.8062 | `models/model_metadata.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Locked Test KS = 0.6427 | `models/model_metadata.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Locked Test Gini = 0.7937 | `models/model_metadata.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Locked Test Brier = 0.0836 | `models/model_metadata.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Score range is 300 to 850 | `models/credit_policy.json`, `src/models/scoring.py` | `README.md`, `MODEL_CARD.md` | PASS |
| Approve threshold = `0.20789473684210527` | `models/credit_policy.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Reject threshold = `0.71875` | `models/credit_policy.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| Risk grade boundaries are frozen PD cutoffs | `models/credit_policy.json` | `README.md`, `MODEL_CARD.md`, `docs/methodology.md` | PASS |
| API endpoints include health, model info, predict, batch predict, monitoring reference, monitoring analyze | `api/main.py` | `README.md`, `docs/api_usage.md` | PASS |
| No `/explain` endpoint is implemented | `api/main.py`, `src/inference/predictor.py` | `README.md`, `MODEL_CARD.md`, `docs/api_usage.md` | PASS |
| Monitoring reference is Development population, not Locked Test | `src/monitoring/monitor.py`, `reports/monitoring/reference_profile.json` | `README.md`, `MODEL_CARD.md`, `docs/architecture.md`, `docs/monitoring.md` | PASS |
| Monitoring PSI thresholds are 0.10 warning and 0.25 alert | `configs/config.yaml`, `reports/monitoring/reference_profile.json` | `README.md`, `docs/monitoring.md` | PASS |
| Dashboard calls FastAPI and does not compute PSI locally | `dashboard/api_client.py`, `dashboard/app.py`, `dashboard/utils.py` | `README.md`, `docs/architecture.md`, `docs/api_usage.md`, `docs/monitoring.md` | PASS |

## Checks

| Check | Result |
| --- | --- |
| Target documentation files exist | PASS |
| Relative documentation links exist | PASS |
| Absolute local path scan for new docs | PASS |
| Current-state docs avoid stale frozen metrics | PASS |
| Current-state docs do not claim production banking approval | PASS |
| Historical planning docs were not rewritten | PASS |

Note: `docs/archive/CREDIT_RISK_PROJECT_PLAN_FULL.md` and `docs/archive/CREDIT_RISK_PRODUCT_OVERVIEW.md` are retained as historical planning documents. They may mention planned endpoints or future scope that are not implemented; the current project-facing documentation now states the implemented API surface explicitly.
