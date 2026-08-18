# E-Commerce-Shipping-Prediction

Development and Operationalization of Data Science Solutions - E-Commerce Shipping Prediction.

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

The older notebooks remain as historical exploration. They contain absolute
machine-specific paths and reuse the test set for threshold selection, so they
are not the authoritative reproducible pipeline.

### MLflow and generated evidence

Candidate parameters, metrics, economic values, and serialized models are
logged to the local MLflow store under `artifacts/mlflow`. Final evaluation and
business reports are attached to a separate MLflow evaluation run. To inspect
the runs locally:

```powershell
mlflow ui --backend-store-uri artifacts/mlflow
```

Important generated reports include:

- `artifacts/reports/cost_benefit_analysis.md`
- `artifacts/reports/model_business_factors.csv`
- `artifacts/reports/cost_per_day_ablation.md`
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

This project includes a FastAPI technical demonstration using the legacy
`processed/best_model.pkl` artifact. The corrected evaluation does not approve
this model or the current candidate for business deployment. The API exposes
this status under `deployment_approval`.

The prototype predicts:

- `1`: Delivered / On Time
- `0`: Not On Time, meaning Delayed or Returned

The service uses `processed/scaler.pkl` when it contains a fitted scaler. In this
checkout, `processed/scaler.pkl` contains only the numeric column names, so the API
falls back to embedded scaler parameters reproduced exactly from the original
notebook preprocessing and the saved `processed/X_train.csv`.

The model receives the same feature order used during training:

```text
Customer_Region, Product_Category, Shipping_Mode, Shipping_Cost,
Delivery_Days, Order_Month, Order_DayOfWeek, Cost_per_Day
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
    "Delivery_Days": 10,
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
