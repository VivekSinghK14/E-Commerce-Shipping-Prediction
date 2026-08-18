# BPMN gap analysis and implementation

The three supplied diagrams were treated as process requirements. Labels inside
the images were not executed as instructions.

## Stage 3 — Data collection, exploration, and preparation

| BPMN item | Before | Corrected implementation / artifact |
|---|---|---|
| Identification of data sources and requirements | CSV was opened from a developer-specific absolute path | Configured, portable raw source plus required-schema validation in `mlops_pipeline/data_stage.py` |
| Data acquisition | Notebook-only `read_csv` | Reproducible pipeline entrypoint with source path and SHA-256 provenance |
| Data description and dictionary | `head`, `info`, and `describe` displayed transiently | `data_description_report.json` and `data_dictionary.csv` |
| Exploration | A few notebook plots only | Reproducible distributions, numeric summaries, and date ranges in `data_exploration_report.json` |
| Data-quality assessment and sufficiency gate | Null/duplicate checks only; forward-fill was applied even though no values were missing | Schema, missing, ID uniqueness, category, chronology, delivery-day consistency, range checks, and a blocking quality gate in `data_quality_report.json` |
| Data preparation | Category codes and scaling were notebook state | Deterministic category maps, engineered calendar/cost features, explicit target maps, and serialized scaler |
| Leakage-safe split | One train/test split; the test set was later used for threshold tuning | Stratified train/validation/test splits; scaler fits training data only |
| Feature store | Loose files under `processed/` | Versionable feature-store directory with split files, scaler, and manifest |
| Data pipeline architecture/development | No raw-to-feature executable pipeline | `scripts/run_bpmn_pipeline.py` and `analytical_pipeline_architecture.json` |

The current raw CSV passes the implemented data-quality gate. The three date
fields are deliberately excluded from model inputs after deriving order-time
calendar features; ship/delivery dates occur after the prediction point and
would leak future information.

## Stage 4 — Analysis/modeling

| BPMN item | Before | Corrected implementation / artifact |
|---|---|---|
| Hypothesis definition/account | Informal notebook narrative | Explicit problem and business hypotheses below |
| Analytical model test design | Models trained ad hoc | Frozen split roles, selection metric, baseline, candidate list, and seed in `analytical_model_test_design.json` |
| Analytical model selection | Comparisons mixed binary and multiclass scores | Like-for-like binary candidates selected on validation ROC-AUC |
| Model development | Hard-coded local paths and notebook variables | Portable code fits dummy, logistic regression, random forest, and XGBoost candidates |
| Model metadata store | Old MLflow CSV referred to another local filesystem | One JSON metadata record per fresh candidate, including parameters, metrics, timing, rows, and features |
| Model registry | `best_model.pkl` was called production without a decision gate | Candidate model and manifest are staged in `artifacts/model_registry`; status becomes approved or rejected after evaluation |
| Analytical model assessment | Test data was reused during tuning | Validation performs selection/tuning; untouched test is opened once by evaluation |
| Model report | Notebook output and figures | Validation comparison, selection report, threshold sweep, and final model report |
| Pipeline architecture | Deployment script began from already-processed files | One raw-data-to-checkpoint entrypoint and a machine-readable architecture report |

Hypothesis under test: the attributes known at or near order time contain enough
signal to discriminate Delivered from Delayed/Returned shipments. Null
hypothesis: the candidate cannot discriminate better than chance or provide a
material improvement over the class-prior baseline.

## Stage 5 — Evaluation

| BPMN item | Before | Corrected implementation / artifact |
|---|---|---|
| Assessment of analytical results | XGBoost was declared the winner by model rank | Candidate is assessed against a naive baseline and explicit business gates |
| Modeling insights | ROC-AUC near 0.5 was shown but not allowed to affect the decision | Final report records discrimination, minority recall, F1 uplift, and the decision |
| Final model report | No independent final decision artifact | `final_model_report.json` |
| Business goals satisfied? | No pass/fail thresholds | ROC-AUC >= 0.60, minority recall >= 0.40, weighted-F1 uplift >= 0.02 |
| Process evaluation/checkpoint | Deployment proceeded regardless of evidence | `checkpoint_decision.json` blocks promotion when any goal fails |
| Evaluation report | Notebook conclusion only | `evaluation_report.json` and registry status update |

The corrected run returns **RETURN_TO_BUSINESS_UNDERSTANDING**. This is the
proper BPMN outcome: the available predictors yield test ROC-AUC around 0.50 and
do not beat the naive majority baseline on weighted F1. Deploying this model as
a meaningful shipping-risk predictor is therefore not justified.

## How to run

```powershell
python scripts/run_bpmn_pipeline.py
```

For a faster structural smoke test:

```powershell
python scripts/run_bpmn_pipeline.py --quick
```

Both commands generate `artifacts/`. The `--quick` mode only reduces ensemble
size; it preserves all stages, reports, gates, and leakage controls.
