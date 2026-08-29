# Model Card - Credit Risk Model

## 1. Model Overview

| Item | Value |
| --- | --- |
| Model purpose | Estimate applicant Probability of Default (PD). |
| Model family | LightGBM |
| Artifact version | `2026-08-29-frozen-notebook-04` |
| Calibration | Isotonic Regression |
| Policy status | `frozen` |
| Primary raw feature count | 18 |
| Engineered feature count | 13 |
| Score range | 300 to 850 |

Frozen artifacts:

- `models/champion_model.joblib`
- `models/calibrated_model.joblib`
- `models/feature_schema.json`
- `models/credit_policy.json`
- `models/model_metadata.json`

## 2. Intended Use

The model is intended for a portfolio demonstration of application credit risk assessment:

- estimate applicant PD
- rank applicants by modeled default risk
- map PD to an internal project credit score
- assign a risk grade
- apply a simulated loan decision policy
- support batch portfolio scoring and monitoring demonstrations

Predictions are decision-support outputs for a portfolio project, not real lending authority.

## 3. Out-of-Scope Use

This model is not intended for:

- autonomous real-world loan approvals or rejections
- regulatory credit approval without independent validation
- fraud detection
- AML or mule detection
- collections strategy
- behavioral early-warning default prediction
- production deployment without governance, monitoring, security, and audit controls

## 4. Dataset

Verified from current frozen metadata and data audit:

| Item | Value |
| --- | ---: |
| Rows | 32,581 |
| Columns | 29 |
| Target | `loan_status` |
| Identifier | `client_ID` |
| Default observations | 7,108 |
| Default rate | 21.82% |
| Development rows | 26,064 |
| Locked Test rows | 6,517 |

Data quality findings:

- `person_emp_length`: 895 missing values, 2.75%.
- `loan_int_rate`: 3,116 missing values, 9.56%.
- `person_age`: 5 values outside `[20, 100]`.
- `person_emp_length`: 2 values violate the employment-length-vs-age business rule.
- No duplicate rows and no duplicate `client_ID` values were found.

## 5. Target Definition

Target column:

```text
loan_status
```

The project treats `loan_status` as a binary default indicator:

- `0`: non-default / good outcome
- `1`: default / bad outcome

The target is never used as a model input.

## 6. Feature Governance

Primary features used by the frozen model:

| Feature |
| --- |
| `person_age` |
| `person_income` |
| `person_home_ownership` |
| `person_emp_length` |
| `loan_intent` |
| `loan_amnt` |
| `cb_person_default_on_file` |
| `cb_person_cred_hist_length` |
| `marital_status` |
| `education_level` |
| `employment_type` |
| `loan_term_months` |
| `loan_to_income_ratio` |
| `other_debt` |
| `debt_to_income_ratio` |
| `open_accounts` |
| `credit_utilization_ratio` |
| `past_delinquencies` |

Excluded and restricted groups:

| Group | Columns | Treatment |
| --- | --- | --- |
| Target | `loan_status` | Never model input. |
| Identifier | `client_ID` | Excluded to avoid memorization. |
| Extended-only leakage candidates | `loan_grade`, `loan_int_rate` | Excluded from Primary Model; used only for research/ablation. |
| Audit-only variables | `gender`, `country`, `state`, `city`, `city_latitude`, `city_longitude` | Reserved for fairness/proxy review; excluded from model input. |
| Redundant excluded variables | `loan_percent_income`, `loan_to_income_ratio_fe`, `total_debt_to_income_ratio` | Excluded from Primary Model. |

The project principle is that predictive signal does not automatically imply modeling eligibility.

## 7. Preprocessing

Current preprocessing is implemented in `src/data/preprocessing.py`:

- `BusinessRuleCleaner`
  - Adds original missing flags such as `person_emp_length_missing_flag`.
  - Sets `person_age` outside `[20, 100]` to missing.
  - Sets invalid `person_emp_length` to missing when negative or greater than `person_age - 14`.
- `CreditRiskFeatureBuilder`
  - Adds approved ratio, flag, and interaction features.
- `build_preprocessor`
  - Numeric median imputation.
  - Categorical most-frequent imputation.
  - One-hot encoding with unknown-category handling.
  - Optional scaling support; disabled for the LightGBM champion.

Preprocessing is inside the sklearn pipeline to avoid pre-split leakage from global imputation or encoding.

## 8. Feature Engineering

Final engineered features:

| Engineered feature |
| --- |
| `person_emp_length_missing_flag` |
| `existing_debt_to_income_ratio` |
| `loan_amount_per_month` |
| `high_loan_to_income_flag` |
| `high_debt_to_income_flag` |
| `high_existing_debt_to_income_flag` |
| `previous_default_flag` |
| `past_delinquency_flag` |
| `thin_credit_history_flag` |
| `high_credit_utilization_flag` |
| `loan_to_income_x_previous_default` |
| `debt_to_income_x_past_delinquency` |
| `credit_history_x_credit_utilization` |

Feature engineering is intentionally interpretable and implemented through reusable source code.

## 9. Training Strategy

Split:

- 80% Development population.
- 20% Locked Test population.
- Random state: 42.
- Stratification: enabled.

Development workflow:

- data audit
- EDA
- leakage audit
- feature-set design
- preprocessing design
- baseline comparison
- LightGBM and CatBoost comparison
- tuning
- calibration
- champion selection
- score, grade, and policy design

Locked Test is reserved for final frozen evaluation and is not used as the monitoring reference.

## 10. Final Model

Frozen champion: LightGBM.

Selected parameters:

| Parameter | Value |
| --- | ---: |
| `colsample_bytree` | 0.85 |
| `learning_rate` | 0.05 |
| `max_depth` | -1 |
| `min_child_samples` | 40 |
| `n_estimators` | 150 |
| `num_leaves` | 31 |
| `subsample` | 0.85 |

The final inference artifact is the calibrated model at `models/calibrated_model.joblib`.

## 11. Calibration

Calibration method: Isotonic Regression.

Calibration is used because the downstream system consumes PD as a probability, not only as a ranking score. PD drives score mapping, risk grades, decision policy, and monitoring.

## 12. Evaluation

Frozen Locked Test metrics:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.8969 |
| PR-AUC | 0.8062 |
| KS | 0.6427 |
| Gini | 0.7937 |
| Brier Score | 0.0836 |

Additional threshold-0.5 metrics are stored in `models/model_metadata.json`, but the lending decision simulation uses separate frozen PD thresholds.

## 13. Credit Score Mapping

The internal project score is calculated from calibrated PD:

```text
score = 575 + 72 * ln((1 - PD) / PD)
```

Score config:

| Item | Value |
| --- | ---: |
| Minimum score | 300 |
| Maximum score | 850 |
| Base score | 575 |
| Factor | 72.0 |
| PD clipping epsilon | 0.000001 |

Higher score means lower modeled PD. This is not FICO or an official bureau score.

## 14. Risk Grades

Risk grades are assigned from calibrated PD using frozen Development-PD boundaries:

| Grade | Meaning | PD interval |
| --- | --- | --- |
| A | Lowest modeled risk | `PD <= 0.011013` |
| B | Lower modeled risk | `0.011013 < PD <= 0.040195` |
| C | Medium modeled risk | `0.040195 < PD <= 0.107527` |
| D | Higher modeled risk | `0.107527 < PD <= 0.364103` |
| E | Highest modeled risk | `PD > 0.364103` |

These are internal project grades, not regulatory grades.

## 15. Decision Policy

Frozen policy:

| Rule | Decision |
| --- | --- |
| `PD < 0.20789473684210527` | `APPROVE` |
| `0.20789473684210527 <= PD < 0.71875` | `MANUAL_REVIEW` |
| `PD >= 0.71875` | `REJECT` |

The policy is a simulated project-level business policy and should not be interpreted as real bank policy.

## 16. Explainability

Current serving status:

```text
risk_drivers = None
explanation_status = "explanation_unavailable"
```

The project has research notes for SHAP and fairness/proxy review, but production-safe risk-driver mapping from encoded model features back to raw business concepts is deferred.

## 17. Fairness and Proxy Risk

Audit-only variables are excluded from model input:

- `gender`
- `country`
- `state`
- `city`
- `city_latitude`
- `city_longitude`

Exclusion reduces direct use of sensitive or geographic proxy variables in the Primary Model, but it does not prove the model is fair or free from proxy effects. Fairness conclusions require a separate governance process.

## 18. Monitoring

Monitoring compares:

```text
Development Reference Population -> Reference Profile
Current Portfolio Batch -> Monitoring Report
```

Implemented signals:

- feature drift
- missingness drift
- PD drift
- credit score drift
- risk grade drift
- decision drift
- optional performance monitoring if realized `loan_status` labels are available
- optional calibration gap when labels are available

Reference profile:

```text
reports/monitoring/reference_profile.json
```

Monitoring is simulated batch monitoring, not real-time production monitoring.

## 19. Limitations

- Portfolio project, not a production-approved banking model.
- Dataset is static and does not provide real production timestamps.
- Monitoring is simulated, not live operational monitoring.
- Decision thresholds are project assumptions.
- Internal score is not FICO.
- Explainability is not production-safe in the inference/API layer.
- Audit-only feature exclusion does not establish fairness.
- Drift does not prove causal performance deterioration.
- Real deployment would require governance, security, audit logging, monitoring operations, and independent validation.

## 20. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m compileall src api dashboard tests
python -m pytest -q
```

Start API:

```bash
python -m uvicorn api.main:app --reload
```

Start dashboard:

```bash
python -m streamlit run dashboard/app.py
```

Monitoring:

```bash
python -m src.monitoring.monitor build-reference
python -m src.monitoring.monitor analyze --input reports/monitoring/synthetic_current_batch_smoke.csv
```

Regenerate artifacts only when intentionally refreshing the frozen model:

```bash
python -m src.models.train
```

## 21. Governance Status

The current model, schema, calibration, score mapping, risk grades, and decision policy are frozen for this portfolio implementation.

This does not mean the model is regulatory-approved, production-approved, or bank-validated.
