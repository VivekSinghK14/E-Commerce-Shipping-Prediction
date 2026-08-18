# Professor requirements traceability

| Requirement | Status | Executable implementation | Evidence |
|---|---|---|---|
| Cost/benefit of right and wrong classifications | Complete | `classification_economics` evaluates TN, TP, FP, and FN using explicit scenario values | `cost_benefit_analysis.json/.md` |
| Business factors from each model | Complete | Every validation candidate records quality, economics, latency, interpretability, and operational complexity | `model_business_factors.csv`; MLflow runs |
| Correct data format at pipeline start and end | Complete | Raw and feature-store contracts are validated; invalid outputs stop the run | `input_data_contract.json`, `output_data_contract.json` |
| Test cases | Complete | API, data stage, contracts/artifacts, business economics, retraining trigger, and checkpoint tests | `tests/`; `pytest -q` |
| Retraining behavior | Complete | Data-change, age, degradation, and manual triggers; versioned runs; gated promotion; champion preservation | `scripts/retrain_model.py`, `mlops_pipeline/retraining.py` |
| Integrated pipelines | Complete | One entrypoint invokes data → modeling → evaluation | `scripts/run_bpmn_pipeline.py` |
| Feature storage in data pipeline | Complete | Train/validation/test features, targets, scaler, and manifest | `artifacts/feature_store/` |
| BPMN and UML diagrams | Complete | BPMN-aligned flow, UML components, and retraining sequence | `docs/architecture.md` |
| Cost_per_Day improvement and interpretation | Complete | Controlled same-model ablation with and without the feature; importance reported separately | `cost_per_day_ablation.json/.md` |
| Business-objective pros and cons | Complete | Explicit benefits, limitations, conclusion, and next action | `business_objective_assessment.json/.md` |
| MLflow integration | Complete | Candidate models/metrics/parameters and final evaluation reports logged to a local tracking store | `artifacts/mlflow/`; run IDs in reports |

## Important conclusion

Implementation completeness does not mean the model should be deployed. The
pipeline is complete, but the current data does not pass predictive-performance
gates. The BPMN decision remains `RETURN_TO_BUSINESS_UNDERSTANDING`.
