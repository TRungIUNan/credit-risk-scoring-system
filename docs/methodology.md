# Methodology

This document explains the Data Science methodology behind the frozen credit risk project. Current frozen artifacts and source code are the source of truth.

## 1. Problem Definition

The project estimates applicant Probability of Default:

```text
PD = P(Default = 1 | applicant information)
```

The downstream decision flow is:

```text
PD -> Internal Credit Score -> Risk Grade -> APPROVE / MANUAL_REVIEW / REJECT
```

The task is credit default risk modeling, not fraud detection, AML, collections, or behavioral early-warning prediction.

## 2. Dataset Audit

Verified audit facts:

| Item | Value |
| --- | ---: |
| Rows | 32,581 |
| Columns | 29 |
| Target | `loan_status` |
| Identifier | `client_ID` |
| Default observations | 7,108 |
| Default rate | 21.82% |

Data quality findings:

- `person_emp_length` has 895 missing values, 2.75%.
- `loan_int_rate` has 3,116 missing values, 9.56%.
- No duplicate rows were found.
- `client_ID` is unique.
- `person_age` has 5 values outside `[20, 100]`.
- `person_emp_length` has 2 age-inconsistent values.
- `loan_grade` and `loan_int_rate` are leakage-risk watchlist variables.

## 3. EDA

The EDA notebook investigates target distribution, numeric and categorical risk patterns, correlations, redundancy, missingness, outliers, leakage risk, and fairness/proxy concerns.

EDA is used to inform feature governance and modeling hypotheses. It is not used to bypass the locked-test protocol.

## 4. Feature Governance

The frozen Primary Model uses 18 application-time features:

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

Excluded from the Primary Model:

- `loan_status`: target label.
- `client_ID`: identifier.
- `loan_grade`: potential post-underwriting risk grade.
- `loan_int_rate`: potential pricing/post-underwriting signal.
- `gender`, `country`, `state`, `city`, `city_latitude`, `city_longitude`: audit-only sensitive/proxy variables.
- `loan_percent_income`, `loan_to_income_ratio_fe`, `total_debt_to_income_ratio`: redundant LTI aliases or non-final redundant definitions.

## 5. Leakage Audit

Leakage means information unavailable at application time accidentally helps the model. Controls used in the project:

- Separate Primary and Extended feature sets.
- Exclude target, ID, leakage-risk variables, audit-only variables, and redundant aliases from the frozen Primary Model.
- Keep preprocessing inside the pipeline.
- Use stratified Development/Locked Test split.
- Use Locked Test only for final frozen evaluation.
- Enforce forbidden fields at inference/API level.

## 6. Split and Cross-Validation

The frozen split is:

| Population | Rows |
| --- | ---: |
| Development | 26,064 |
| Locked Test | 6,517 |

Split settings:

- Test size: 0.20.
- Random state: 42.
- Stratification: enabled.
- Development CV reference: 5-fold stratified cross-validation.

Out-of-fold predictions mean each development observation is scored by a model that was not trained on that observation. This supports less biased development-time comparisons and policy design.

## 7. Preprocessing

Current reusable preprocessing is implemented in `src/data/preprocessing.py`:

- `BusinessRuleCleaner`
  - Adds original missing indicator flags such as `person_emp_length_missing_flag`.
  - Sets invalid `person_age` outside `[20, 100]` to missing.
  - Sets invalid `person_emp_length` values to missing when negative or greater than `person_age - 14`.
- `CreditRiskFeatureBuilder`
  - Adds approved ratio, flag, and interaction features.
- `build_preprocessor`
  - Numeric median imputation.
  - Optional numeric scaling, disabled for the LightGBM champion.
  - Categorical most-frequent imputation.
  - One-hot encoding with unknown-category handling.

Preprocessing is inside sklearn pipelines rather than applied globally before splitting.

## 8. Feature Engineering

Final engineered features stored in `models/feature_schema.json`:

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

The feature engineering is intentionally interpretable and hypothesis-driven.

## 9. Model Research Flow

The notebook flow documents:

1. Dummy baseline.
2. Logistic Regression.
3. LightGBM.
4. CatBoost.
5. Model comparison.
6. Hyperparameter tuning.
7. Probability calibration.
8. Champion selection.
9. Credit score, risk grade, and decision policy.
10. Locked Test evaluation.
11. SHAP, fairness/proxy review, and stress testing notes.

## 10. Final Model

Frozen champion:

| Item | Value |
| --- | --- |
| Model | LightGBM |
| Artifact version | `2026-08-29-frozen-notebook-04` |
| Calibration | Isotonic Regression |
| Policy status | `frozen` |

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

## 11. Calibration

The final model uses isotonic calibration. Calibration aligns predicted PD values with observed default frequencies, which matters because PD drives score mapping, grade assignment, policy decisions, and monitoring.

## 12. Locked Test Evaluation

Frozen Locked Test metrics:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.8969 |
| PR-AUC | 0.8062 |
| KS | 0.6427 |
| Gini | 0.7937 |
| Brier Score | 0.0836 |

Confusion-matrix-derived fields at threshold 0.5 are stored in metadata, but policy decisions use separate PD thresholds rather than a single classification threshold.

## 13. Credit Policy

Credit score mapping:

```text
score = 575 + 72 * ln((1 - PD) / PD)
```

The score is clipped to `[300, 850]`. Higher score means lower modeled PD. It is an internal project score, not FICO.

Risk grades use Development calibrated-PD quantile boundaries:

| Boundary | PD cutoff |
| --- | ---: |
| A/B | 0.011013 |
| B/C | 0.040195 |
| C/D | 0.107527 |
| D/E | 0.364103 |

Decision thresholds:

| Rule | Decision |
| --- | --- |
| `PD < 0.20789473684210527` | `APPROVE` |
| `0.20789473684210527 <= PD < 0.71875` | `MANUAL_REVIEW` |
| `PD >= 0.71875` | `REJECT` |

These thresholds are simulated project policy assumptions.

## 14. Policy Freeze

The frozen system is represented by:

- `models/calibrated_model.joblib`
- `models/feature_schema.json`
- `models/credit_policy.json`
- `models/model_metadata.json`

Task 7 documentation does not modify the frozen model, thresholds, schema, or policy.

## 15. Monitoring

Monitoring compares:

```text
Development Reference Population -> Reference Profile
Current Portfolio Batch -> Monitoring Report
```

It reports feature drift, missingness drift, PD drift, score drift, risk grade drift, decision drift, and optional performance/calibration metrics when realized labels exist.

Monitoring is a simulation because the dataset has no true production timeline.

## 16. Engineering Reproducibility

Implemented interfaces:

```bash
python -m src.models.train
python -m src.inference.predictor
python -m src.monitoring.monitor build-reference
python -m src.monitoring.monitor analyze --input reports/monitoring/synthetic_current_batch_smoke.csv
python -m uvicorn api.main:app --reload
python -m streamlit run dashboard/app.py
python -m pytest -q
```

`python -m src.models.train` regenerates frozen artifacts and should only be run intentionally.

## 17. Limitations

- The dataset is static and lacks a real production timeline.
- The model is not approved for real autonomous lending decisions.
- Feature availability assumptions are project-level assumptions.
- Fairness/proxy audit variables are excluded from model input, but this does not prove fairness.
- Risk-driver explanations are not production-safe in the inference layer.
- Monitoring drift signals require investigation and do not automatically justify retraining.
