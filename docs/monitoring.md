# Monitoring

The monitoring layer is a model monitoring simulation for the frozen credit risk portfolio project. It compares a Development reference population with a current applicant batch. It is not live production monitoring and does not use a real production timeline.

## Reference Population

The reference population is the Development side of the frozen 80/20 split:

- Split source: `configs/config.yaml`.
- Test size: 0.20.
- Random state: 42.
- Stratification: enabled.
- Reference rows: 26,064.
- Locked Test is not used as the monitoring reference.

The reference profile is saved to:

```text
reports/monitoring/reference_profile.json
```

## Reference Profile

The reference profile stores aggregate monitoring statistics only:

- reference row count
- creation timestamp
- model version
- schema version
- policy status
- calibration method
- feature list
- numeric bin edges
- numeric reference proportions
- categorical reference proportions
- missing rates
- PD reference distribution
- internal credit score reference distribution
- risk grade distribution
- decision distribution

It does not store raw Development records.

## Numeric PSI

Population Stability Index is implemented as:

```text
PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
```

The implementation uses epsilon smoothing to avoid `log(0)`.

## Numeric Bins

Numeric bins are fitted on reference data only. Current batches are assigned to the stored reference bin edges.

This is critical: current data never refits numeric bin edges. If the current distribution shifts, the same reference bins are still used.

The current default number of numeric bins is 10.

## Categorical PSI

Categorical monitoring uses categories observed in the reference population.

Current values are mapped as follows:

- Known reference category: kept as-is.
- Unseen current category: `__OTHER__`.
- Missing value: `__MISSING__`.

## PSI Thresholds

Current monitoring config:

| PSI range | Status |
| --- | --- |
| `< 0.10` | `STABLE` |
| `0.10 <= PSI < 0.25` | `WARNING` |
| `>= 0.25` | `ALERT` |

These thresholds are configurable heuristics, not universal regulatory standards.

## Feature Drift

All 18 frozen Primary Features are monitored:

- numeric or categorical PSI
- reference missing rate
- current missing rate
- missing rate delta
- status

Rows are sorted by severity and then by PSI descending.

## Missingness Drift

Missingness drift is reported separately from PSI. It is a data-quality signal and does not automatically mean the model has failed.

## Model Output Drift

The current batch is scored through `CreditRiskPredictor`, then output distributions are monitored:

- PD drift
- internal credit score drift
- risk grade drift
- decision drift

Monitoring does not load or score the model independently.

## PD Drift

PD drift uses the stored reference PD binning and reports:

- `pd_psi`
- `pd_status`
- reference/current mean PD
- reference/current median PD
- reference/current PD bucket distributions

## Credit Score Drift

Credit score drift reports:

- `score_psi`
- `score_status`
- reference/current mean score
- reference/current median score
- reference/current score bucket distributions

Higher or lower score shifts are reported as distribution shifts, not automatically assigned a cause.

## Risk Grade Drift

Risk grade drift is categorical PSI over:

```text
A, B, C, D, E
```

Grades are frozen project grades, not regulatory grades.

## Decision Drift

Decision drift is categorical PSI over:

```text
APPROVE, MANUAL_REVIEW, REJECT
```

It also reports approval, manual-review, and rejection rates for reference and current populations.

## Performance Monitoring

Labels are optional. If a current batch does not include realized `loan_status`, the report sets:

```text
performance_status = labels_unavailable
```

If valid labels are supplied separately, monitoring reports:

- ROC-AUC
- PR-AUC
- KS
- Gini
- Brier Score
- observed default rate
- mean predicted PD
- calibration gap
- absolute calibration gap

Label guards:

- length must match current batch rows
- labels must be binary 0/1
- labels must include both classes for AUC-style metrics
- minimum labeled rows: 20

## Calibration Monitoring

Calibration monitoring is intentionally simple:

```text
calibration_gap = mean_predicted_pd - observed_default_rate
```

This is not a replacement for a full calibration governance process.

## Overall Status

Overall status is the worst severity across feature drift and output drift:

- any `ALERT` component -> `ALERT`
- else any `WARNING` component -> `WARNING`
- else `STABLE`

## CLI

Build reference:

```bash
python -m src.monitoring.monitor build-reference
```

Analyze a current CSV:

```bash
python -m src.monitoring.monitor analyze --input reports/monitoring/synthetic_current_batch_smoke.csv
```

If `loan_status` exists in the CSV, the CLI separates it into labels before prediction.

## API and Dashboard

FastAPI endpoints:

- `GET /monitoring/reference`
- `POST /monitoring/analyze`

Dashboard behavior:

- Upload current portfolio CSV.
- Validate required applicant fields.
- Separate optional `loan_status`.
- Call FastAPI.
- Display summary, feature drift table, output drift charts, optional performance metrics, and JSON download.

Dashboard does not calculate PSI locally.

## Limitations

- Monitoring is simulated batch monitoring.
- The dataset has no real production timestamp stream.
- The reference is the Development population, not a live production baseline.
- PSI signals distribution shift, not causal model degradation.
- Performance monitoring requires realized labels.
- Retraining decisions require separate validation and governance.
