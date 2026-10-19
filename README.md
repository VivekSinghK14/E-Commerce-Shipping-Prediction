# E-Commerce-Shipping-Prediction

Development and Operationalization of Data Science Solutions - E-Commerce Shipping Prediction.

## Problem scope: Delivered vs Delayed

The model answers one binary question: **will an order be Delivered (1) or
Delayed (0)?** The raw CSV also contains `Returned` orders. Returns are a
different business process (reverse logistics) and are **removed completely**
at the very first pipeline step, before description, quality checks, splitting,
modeling, evaluation, or serving.

| Source outcome | Rows | Used? |
|---|---:|---|
| Delivered | 40,062 | yes, class 1 |
| Delayed | 7,505 | yes, class 0 |
| Returned | 2,433 | no, removed at ingestion |
| **Modeled rows** | **47,567** | 60% train / 20% validation / 20% test |

The removal is recorded in `data_preparation_report.json` and
`data_quality_report.json` (`scope_filter`), and enforced by tests.

## BPMN stages 3–5 (authoritative workflow)

The reproducible workflow from the raw `50K Records.csv` through the evaluation
checkpoint is:

```powershell
python scripts/run_bpmn_pipeline.py
```

For the complete six-stage lifecycle required by the project proposal, use:

```powershell
python scripts/run_end_to_end_pipeline.py
```

The complete plain-language implementation report is generated at
`output/pdf/ecommerce_mlops_complete_project_report.pdf` by:

```powershell
python scripts/generate_project_report.py
```

Use `--quick` for a faster smoke run. Generated data reports, feature-store
files, model metadata/registry records, model reports, and the evaluation
checkpoint are written under `artifacts/`.

The current evidence does **not** satisfy the configured business gates. The
correct checkpoint is `RETURN_TO_BUSINESS_UNDERSTANDING`; the candidate is not
promoted. See [the BPMN gap analysis](docs/bpmn_gap_analysis.md) for the exact
diagram-to-code mapping and explanation.

The project's final selected algorithm is **XGBoost** (highest validation
ROC-AUC among the non-baseline candidates). Its status is
`SELECTED_FOR_PROTOTYPE_NOT_APPROVED_FOR_DEPLOYMENT`: untouched-test ROC-AUC is
about 0.49 (chance level), so evaluation does not approve it. It is
nevertheless **deployed through a recorded management override** for the
course demonstration; see [Deployment by management override](#deployment-by-management-override).

### Notebooks

The notebooks walk through the same Delivered-vs-Delayed process and read the
pipeline's feature store, so their numbers match the reports. Run the pipeline
first, then the notebooks in this order:

1. `E- commerce.ipynb`: data understanding, Returned removal, EDA, leakage review
2. `modelling.ipynb/Models.ipynb`: candidate training and validation-only selection
3. `evalaution/stage4_evaluation.ipynb`: untouched-test evaluation and checkpoint

### MLflow and generated evidence

Candidate parameters, metrics, economic values, and serialized models are
logged to the local MLflow store under `artifacts/mlflow`. Final evaluation and
business reports are attached to a separate MLflow evaluation run. To inspect
the runs locally:

```powershell
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/tracking.db
```

Important generated reports include:

- `artifacts/reports/naive_vs_best_candidate.png`
- `artifacts/reports/cost_benefit_analysis.md`
- `artifacts/reports/model_business_factors.csv`
- `artifacts/reports/cost_per_day_leakage_review.md`
- `artifacts/reports/business_objective_assessment.md`
- `artifacts/reports/input_data_contract.json`
- `artifacts/reports/output_data_contract.json`

### Retraining

Retraining is versioned and guarded by the same data, model, evaluation, and
business gates:

```powershell
python scripts/retrain_model.py --force
```

Automatic triggers include changed source-data hash, model age, and observed
ROC-AUC below the configured gate. A candidate replaces the champion only when
every gate passes; otherwise the existing champion is preserved. See
[the BPMN and UML architecture diagrams](docs/architecture.md).

## Deployment

**Live demo:** https://huggingface.co/spaces/VivekSinghK14/shipping-delay-predictor

The live demo is a free static Hugging Face Space. The deployed XGBoost
champion is exported by `python scripts/export_browser_model.py` to
`deployment/static_space/model.json` and evaluated in the browser by
`predictor.js`; `tests/test_browser_model.py` checks that its predictions match
the Python model on all 47,567 in-scope orders. Republish after retraining:

```powershell
python scripts/export_browser_model.py
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='VivekSinghK14/shipping-delay-predictor', repo_type='space', folder_path='deployment/static_space')"
```

The full FastAPI service (with server-side monitoring) runs locally or in
Docker as described below.

## Deployment by management override

The evaluation result is unchanged: the model **failed** its business gates
(`RETURN_TO_BUSINESS_UNDERSTANDING`). Because the course requires a deployed,
demonstrable prototype, the project owner recorded an explicit, audited
deployment override:

```powershell
python scripts/deploy_with_override.py --actor "Vivek Kumar Singh (project owner)" `
  --reason "Course requirement: the prototype must be deployed and demonstrated end to end even though the evaluation gates failed."
```

The override:

- copies the exact evaluated candidate bundle (SHA-256 checked) to `champion_bundle.joblib`;
- writes `champion_manifest.json` with `promotion_status = approved_by_override`, the actor, rationale, failed gates, and test metrics;
- writes `artifacts/reports/deployment_override.json` and appends to `artifacts/lifecycle_decision_log.jsonl`;
- sets the lifecycle status to `DEPLOYED_BY_OVERRIDE_EVALUATION_GATES_FAILED`.

The API then serves predictions, and every response contains
`"deployment_basis": "DEPLOYED_BY_MANAGEMENT_OVERRIDE"` plus a warning that
the model failed evaluation. Without an override, prediction endpoints return
HTTP 503. To withdraw the deployment:

```powershell
python scripts/manage_lifecycle.py discontinue --actor "<name>" --reason "<why>"
```

### Serving component

This project includes a FastAPI serving implementation backed only by an
approved registry bundle (passing evaluation or recorded override). The current
model is served under the override described above.

The prototype predicts:

- `1`: Delivered
- `0`: Delayed

The registered bundle contains the fitted scaler, category mappings, feature
order, decision threshold, and trained estimator.

The model receives the same feature order used during training:

```text
Customer_Region, Product_Category, Shipping_Mode, Shipping_Cost,
Order_Month, Order_DayOfWeek
```

### Run Locally

Create a fresh environment because the committed `.venv` may point to another machine:

```powershell
python -m venv .deploy-venv
.\.deploy-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn deployment.app:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

### Example Prediction

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{
    "Customer_Region": "West",
    "Product_Category": "Home",
    "Shipping_Mode": "Standard",
    "Shipping_Cost": 122,
    "Order_Date": "2024-05-12"
  }'
```

### Docker

```powershell
docker build -t ecommerce-shipping-prediction .
docker run --rm -p 8000:8000 ecommerce-shipping-prediction
```

### Endpoints

- `GET /health`: service and artifact status
- `POST /predict`: one raw shipment record
- `POST /predict/batch`: multiple raw shipment records
- `POST /predict/processed`: one already-encoded/scaled record
- `POST /predict/processed/batch`: multiple already-encoded/scaled records
- `GET /model-info`: selected model and artifact metadata
- `GET /pipeline/status`: prototype MLOps component status
- `GET /monitoring`: prediction and error monitoring summary
- `GET /metrics`: Prometheus-style deployment metrics

Category mappings follow the original notebook's pandas category codes:

- `Customer_Region`: Central=0, East=1, North=2, South=3, West=4
- `Product_Category`: Beauty=0, Electronics=1, Fashion=2, Grocery=3, Home=4, Sports=5
- `Shipping_Mode`: Express=0, Same Day=1, Standard=2

Full deployment documentation, including the PDF requirement traceability matrix,
impact assessment, deployment plan, test plan, monitoring plan, and maintenance
plan, is available in `docs/deployment.md`.
