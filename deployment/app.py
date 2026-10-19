import json
import os
from functools import lru_cache
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from mlops_pipeline.config import FEATURE_COLUMNS, NUMERIC_COLUMNS
from mlops_pipeline.feature_engineering import build_order_time_features
from deployment import operations_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))).resolve()
REPORTS_DIR = ARTIFACTS_DIR / "reports"
REGISTRY_DIR = ARTIFACTS_DIR / "model_registry"
FEATURE_STORE_DIR = ARTIFACTS_DIR / "feature_store"
MODEL_PATH = REGISTRY_DIR / "candidate_model.joblib"
SCALER_PATH = FEATURE_STORE_DIR / "scaler.joblib"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison_validation.csv"
MLFLOW_EXPORT_PATH = ARTIFACTS_DIR / "mlflow"
THRESHOLD_SWEEP_PATH = REPORTS_DIR / "threshold_sweep_validation.csv"
BASELINE_COMPARISON_PATH = REPORTS_DIR / "naive_vs_best_candidate.png"
SELECTION_PATH = REPORTS_DIR / "analytical_model_selection.json"
CHECKPOINT_PATH = REPORTS_DIR / "checkpoint_decision.json"
REGISTRY_MANIFEST_PATH = REGISTRY_DIR / "candidate_manifest.json"
CHAMPION_MANIFEST_PATH = REGISTRY_DIR / "champion_manifest.json"
SYSTEM_STATE_PATH = ARTIFACTS_DIR / "system_state.json"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", str(RUNTIME_DIR / "prediction_log.jsonl"))
)

TARGET_LABELS = {
    0: "Delayed",
    1: "Delivered",
}

OVERRIDE_STATUS = "approved_by_override"
OVERRIDE_WARNING = (
    "Prototype deployment: this model did not meet all evaluation gates (test ROC-AUC "
    "near 0.5). Predictions demonstrate the MLOps pipeline and are not yet a reliable "
    "delay forecast."
)

SERVICE_STARTED_AT = datetime.now(timezone.utc)
MONITORING_LOCK = Lock()
RECENT_PREDICTIONS: deque[dict[str, Any]] = deque(maxlen=20)
MONITORING_STATE: dict[str, Any] = {
    "request_count": 0,
    "prediction_count": 0,
    "batch_request_count": 0,
    "error_count": 0,
    "class_counts": Counter(),
    "probability_sum": 0.0,
    "probability_count": 0,
    "last_prediction_at": None,
}


class ShipmentRecord(BaseModel):
    Customer_Region: str | int = Field(
        ..., examples=["East"], description="Raw label or encoded category code."
    )
    Product_Category: str | int = Field(
        ..., examples=["Electronics"], description="Raw label or encoded category code."
    )
    Shipping_Mode: str | int = Field(
        ..., examples=["Standard"], description="Raw label or encoded category code."
    )
    Shipping_Cost: float = Field(..., examples=[122.0])
    Order_Date: str | None = Field(
        default=None,
        examples=["2024-05-12"],
        description="Used to derive month/day-of-week when those fields are omitted.",
    )
    Order_Month: int | None = Field(default=None, ge=1, le=12)
    Order_DayOfWeek: int | None = Field(
        default=None,
        ge=0,
        le=6,
        description="Monday=0, Sunday=6, matching pandas dt.dayofweek.",
    )


class ProcessedRecord(BaseModel):
    Customer_Region: int
    Product_Category: int
    Shipping_Mode: int
    Shipping_Cost: float
    Order_Month: float
    Order_DayOfWeek: float


class BatchShipmentRequest(BaseModel):
    records: list[ShipmentRecord]


class BatchProcessedRequest(BaseModel):
    records: list[ProcessedRecord]


class OutcomeFeedback(BaseModel):
    prediction_id: str
    actual_delivered: bool
    realized_value_eur: float | None = None


class SupportTicketRequest(BaseModel):
    category: str = Field(..., min_length=2, max_length=80)
    description: str = Field(..., min_length=5, max_length=2000)


class SupportResolution(BaseModel):
    resolution: str = Field(..., min_length=5, max_length=2000)


@lru_cache(maxsize=4)
def _load_bundle(bundle_path: str) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Registered model feature contract does not match the serving API")
    return bundle

app = FastAPI(
    title="E-Commerce Shipping Prediction API",
    version="1.0.0",
    description="Predicts whether an e-commerce shipment will be Delivered or Delayed.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def record_error(error_type: str) -> None:
    with MONITORING_LOCK:
        MONITORING_STATE["error_count"] += 1
    write_jsonl(
        {
            "timestamp": utc_now(),
            "event_type": "error",
            "error_type": error_type,
        }
    )


def write_jsonl(event: dict[str, Any]) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def record_prediction_event(endpoint: str, input_count: int, results: list[dict[str, Any]]) -> None:
    timestamp = utc_now()
    labels = [result["prediction_label"] for result in results]
    probabilities = [
        result["on_time_probability"]
        for result in results
        if result["on_time_probability"] is not None
    ]
    event = {
        "timestamp": timestamp,
        "event_type": "prediction",
        "endpoint": endpoint,
        "input_count": input_count,
        "predictions": labels,
        "avg_on_time_probability": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
        "threshold": results[0]["threshold"] if results else None,
    }

    with MONITORING_LOCK:
        MONITORING_STATE["request_count"] += 1
        MONITORING_STATE["prediction_count"] += len(results)
        MONITORING_STATE["batch_request_count"] += int(input_count > 1)
        MONITORING_STATE["class_counts"].update(labels)
        MONITORING_STATE["probability_sum"] += sum(probabilities)
        MONITORING_STATE["probability_count"] += len(probabilities)
        MONITORING_STATE["last_prediction_at"] = timestamp
        RECENT_PREDICTIONS.append(event)

    write_jsonl(event)
    operations_store.record_predictions(endpoint, results)


def monitoring_snapshot() -> dict[str, Any]:
    with MONITORING_LOCK:
        probability_count = MONITORING_STATE["probability_count"]
        average_probability = (
            MONITORING_STATE["probability_sum"] / probability_count
            if probability_count
            else None
        )
        snapshot = {
            "service_started_at": SERVICE_STARTED_AT.isoformat(),
            "uptime_seconds": round(
                (datetime.now(timezone.utc) - SERVICE_STARTED_AT).total_seconds(), 3
            ),
            "request_count": MONITORING_STATE["request_count"],
            "prediction_count": MONITORING_STATE["prediction_count"],
            "batch_request_count": MONITORING_STATE["batch_request_count"],
            "error_count": MONITORING_STATE["error_count"],
            "class_counts": dict(MONITORING_STATE["class_counts"]),
            "average_on_time_probability": average_probability,
            "last_prediction_at": MONITORING_STATE["last_prediction_at"],
            "prediction_log_path": str(PREDICTION_LOG_PATH),
            "recent_predictions": list(RECENT_PREDICTIONS),
        }
    snapshot["utilization"] = operations_store.utilization_summary()
    return snapshot


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def deployment_approval() -> dict[str, Any]:
    if SYSTEM_STATE_PATH.exists():
        system_state = json.loads(SYSTEM_STATE_PATH.read_text(encoding="utf-8"))
        if system_state.get("status") == "DISCONTINUED":
            return {
                "approved": False,
                "decision": "SYSTEM_DISCONTINUED",
                "serving_mode": "disabled",
                "reason": system_state.get("reason"),
            }

    checkpoint = (
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if CHECKPOINT_PATH.exists()
        else {}
    )
    decision = checkpoint.get("checkpoint_decision", "NO_EVALUATION_CHECKPOINT")
    champion = (
        json.loads(CHAMPION_MANIFEST_PATH.read_text(encoding="utf-8"))
        if CHAMPION_MANIFEST_PATH.exists()
        else {}
    )
    if champion.get("promotion_status") in {"approved", OVERRIDE_STATUS}:
        bundle_path = REGISTRY_DIR / champion.get("bundle_file", "champion_bundle.joblib")
        if bundle_path.exists():
            overridden = champion["promotion_status"] == OVERRIDE_STATUS
            return {
                "approved": True,
                "decision": "DEPLOYED_BY_MANAGEMENT_OVERRIDE" if overridden else "APPROVED_CHAMPION",
                "serving_mode": "override_champion" if overridden else "champion",
                "bundle_path": str(bundle_path),
                "model_name": champion.get("selected_model"),
                "threshold": champion.get(
                    "threshold", champion.get("validation_selected_threshold")
                ),
                "evaluation_decision": decision,
                "override": champion.get("override") if overridden else None,
                "warning": OVERRIDE_WARNING if overridden else None,
            }

    manifest = (
        json.loads(REGISTRY_MANIFEST_PATH.read_text(encoding="utf-8"))
        if REGISTRY_MANIFEST_PATH.exists()
        else {}
    )
    bundle_path = REGISTRY_DIR / manifest.get("bundle_file", "candidate_bundle.joblib")
    approved = decision == "PROCEED_TO_DEPLOYMENT" and bundle_path.exists()
    return {
        "approved": approved,
        "decision": decision,
        "serving_mode": "approved_candidate" if approved else "blocked",
        "bundle_path": str(bundle_path) if approved else None,
        "model_name": manifest.get("selected_model"),
        "threshold": manifest.get("validation_selected_threshold"),
        "warning": (
            None
            if approved
            else "Prediction serving is blocked until an evaluated registry bundle is approved."
        ),
    }


def get_serving_bundle() -> dict[str, Any]:
    approval = deployment_approval()
    if not approval["approved"]:
        raise HTTPException(status_code=503, detail=approval["warning"] or approval["decision"])
    try:
        return _load_bundle(approval["bundle_path"])
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Approved model bundle unavailable: {exc}") from exc


def selected_model_metadata() -> dict[str, Any]:
    selection = (
        json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if SELECTION_PATH.exists()
        else {}
    )
    checkpoint = (
        json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if CHECKPOINT_PATH.exists()
        else {}
    )
    return {
        "model_type": selection.get("selected_model"),
        "selected_model": selection.get("selected_model"),
        "validation_metrics": selection.get("validation_metrics_at_default_threshold"),
        "configured_threshold": selection.get("validation_selected_threshold"),
        "selection_role": selection.get("selection_role"),
        "best_experimental_candidate": selection.get("selected_model"),
        "final_selected_model": checkpoint.get("final_selected_model"),
        "deployment_model": checkpoint.get("deployment_model"),
        "mlflow_runs_available": MLFLOW_EXPORT_PATH.exists(),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    record_error(f"http_{exc.status_code}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError):
    record_error("validation")
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors()}))


def canonicalize_category(name: str, value: str | int) -> int:
    mapping = get_serving_bundle()["category_maps"][name]

    if isinstance(value, int):
        if value in mapping.values():
            return value
        raise HTTPException(
            status_code=422,
            detail=f"{name} code must be one of {sorted(mapping.values())}",
        )

    normalized = " ".join(str(value).strip().split()).casefold()
    reverse = {key.casefold(): code for key, code in mapping.items()}
    if normalized in reverse:
        return reverse[normalized]

    allowed = ", ".join(mapping.keys())
    raise HTTPException(status_code=422, detail=f"{name} must be one of: {allowed}")


def parse_order_date(value: str) -> datetime:
    try:
        return pd.to_datetime(value, format="mixed").to_pydatetime()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Order_Date could not be parsed. Use ISO format such as 2024-05-12.",
        ) from exc


def prepare_raw_frame(records: list[ShipmentRecord]) -> pd.DataFrame:
    bundle = get_serving_bundle()
    rows = [record.model_dump() for record in records]
    for row in rows:
        for name in ["Customer_Region", "Product_Category", "Shipping_Mode"]:
            value = row[name]
            mapping = bundle["category_maps"][name]
            if isinstance(value, int):
                label = next((label for label, code in mapping.items() if code == value), None)
                if label is None:
                    raise HTTPException(status_code=422, detail=f"Invalid {name} code: {value}")
                row[name] = label
    try:
        frame = build_order_time_features(pd.DataFrame(rows), bundle["category_maps"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    frame[NUMERIC_COLUMNS] = scale_numeric_columns(frame[NUMERIC_COLUMNS], bundle)
    return frame


def scale_numeric_columns(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    scaler = bundle["scaler"]
    return pd.DataFrame(scaler.transform(frame), columns=NUMERIC_COLUMNS, index=frame.index)


def prepare_processed_frame(records: list[ProcessedRecord]) -> pd.DataFrame:
    rows = [record.model_dump() for record in records]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def predict_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if list(frame.columns) != FEATURE_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"Expected columns in this order: {FEATURE_COLUMNS}",
        )

    approval = deployment_approval()
    bundle = get_serving_bundle()
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        probabilities = proba[:, positive_index]
        predictions = (probabilities >= threshold).astype(int)
    else:
        predictions = model.predict(frame).astype(int)

    results = []
    for idx, prediction in enumerate(predictions):
        probability = None if probabilities is None else float(probabilities[idx])
        prediction_int = int(prediction)
        results.append(
            {
                "prediction": prediction_int,
                "prediction_id": str(uuid4()),
                "prediction_label": TARGET_LABELS.get(prediction_int, str(prediction_int)),
                "on_time_probability": probability,
                "threshold": threshold if probabilities is not None else None,
                "deployment_basis": approval["decision"],
                "warning": approval.get("warning"),
            }
        )
    return results


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Browser front end for domain users (prediction form, status, monitoring)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    approval = deployment_approval()
    return {
        "status": "ok",
        "model_ready": approval["approved"],
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "monitoring_log_path": str(PREDICTION_LOG_PATH),
        "features": FEATURE_COLUMNS,
        "threshold": approval.get("threshold"),
        "deployment_approval": approval,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    approval = deployment_approval()
    return {
        "target": TARGET_LABELS,
        "features": FEATURE_COLUMNS,
        "artifacts": {
            "model": artifact_status(MODEL_PATH),
            "candidate_bundle": artifact_status(REGISTRY_DIR / "candidate_bundle.joblib"),
            "champion_bundle": artifact_status(REGISTRY_DIR / "champion_bundle.joblib"),
            "scaler": artifact_status(SCALER_PATH),
            "model_comparison": artifact_status(MODEL_COMPARISON_PATH),
            "mlflow_export": artifact_status(MLFLOW_EXPORT_PATH),
            "threshold_sweep": artifact_status(THRESHOLD_SWEEP_PATH),
            "baseline_comparison": artifact_status(BASELINE_COMPARISON_PATH),
        },
        "metadata": selected_model_metadata(),
        "deployment_approval": approval,
    }


@app.get("/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    feature_store_artifacts = {
        "X_train": artifact_status(FEATURE_STORE_DIR / "X_train.csv"),
        "X_validation": artifact_status(FEATURE_STORE_DIR / "X_validation.csv"),
        "X_test": artifact_status(FEATURE_STORE_DIR / "X_test.csv"),
        "manifest": artifact_status(FEATURE_STORE_DIR / "feature_manifest.json"),
    }
    approval = deployment_approval()
    return {
        "status": "ready" if approval["approved"] else "blocked_by_checkpoint",
        "deployment_approval": approval,
        "components": {
            "source_code_repository": "current Git repository",
            "feature_store": feature_store_artifacts,
            "model_registry": {
                "candidate_model": artifact_status(MODEL_PATH),
                "candidate_bundle": artifact_status(REGISTRY_DIR / "candidate_bundle.joblib"),
                "champion_bundle": artifact_status(REGISTRY_DIR / "champion_bundle.joblib"),
            },
            "model_metadata_store": {
                "model_comparison": artifact_status(MODEL_COMPARISON_PATH),
                "mlflow_export": artifact_status(MLFLOW_EXPORT_PATH),
            },
            "serving_component": {
                "framework": "FastAPI",
                "endpoints": ["/health", "/predict", "/predict/batch", "/model-info"],
            },
            "monitoring_component": {
                "endpoints": ["/monitoring", "/metrics"],
                "prediction_log": str(PREDICTION_LOG_PATH),
            },
            "ml_production_pipeline": {
                "script": str(PROJECT_ROOT / "scripts" / "run_production_pipeline.py"),
                "quick_validation_command": (
                    "python scripts/run_production_pipeline.py --quick"
                ),
                "promotion_requires_evaluation_approval": True,
            },
        },
    }


@app.get("/monitoring")
def monitoring() -> dict[str, Any]:
    return monitoring_snapshot()


@app.get("/utilization")
def utilization() -> dict[str, Any]:
    return operations_store.utilization_summary()


@app.post("/feedback")
def submit_feedback(feedback: OutcomeFeedback) -> dict[str, Any]:
    try:
        operations_store.record_outcome(
            feedback.prediction_id,
            feedback.actual_delivered,
            feedback.realized_value_eur,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return operations_store.utilization_summary()


@app.post("/support", status_code=201)
def open_support_ticket(ticket: SupportTicketRequest) -> dict[str, str]:
    ticket_id = operations_store.create_support_ticket(ticket.category, ticket.description)
    return {"ticket_id": ticket_id, "status": "open"}


@app.post("/support/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str, resolution: SupportResolution) -> dict[str, str]:
    try:
        operations_store.resolve_support_ticket(ticket_id, resolution.resolution)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ticket_id": ticket_id, "status": "resolved"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    snapshot = monitoring_snapshot()
    class_counts = snapshot["class_counts"]
    lines = [
        "# HELP ecommerce_prediction_requests_total Prediction endpoint requests.",
        "# TYPE ecommerce_prediction_requests_total counter",
        f"ecommerce_prediction_requests_total {snapshot['request_count']}",
        "# HELP ecommerce_predictions_total Total predictions served.",
        "# TYPE ecommerce_predictions_total counter",
        f"ecommerce_predictions_total {snapshot['prediction_count']}",
        "# HELP ecommerce_prediction_errors_total Total API validation/runtime errors.",
        "# TYPE ecommerce_prediction_errors_total counter",
        f"ecommerce_prediction_errors_total {snapshot['error_count']}",
        "# HELP ecommerce_average_on_time_probability Running average predicted on-time probability.",
        "# TYPE ecommerce_average_on_time_probability gauge",
        "ecommerce_average_on_time_probability "
        f"{snapshot['average_on_time_probability'] or 0.0}",
    ]
    for label, count in class_counts.items():
        lines.append(
            'ecommerce_predictions_by_label_total{label="'
            + label.replace('"', '\\"')
            + '"} '
            + str(count)
        )
    return "\n".join(lines) + "\n"


@app.post("/predict")
def predict(record: ShipmentRecord) -> dict[str, Any]:
    result = predict_frame(prepare_raw_frame([record]))[0]
    record_prediction_event("/predict", 1, [result])
    return result


@app.post("/predict/batch")
def predict_batch(request: BatchShipmentRequest) -> dict[str, list[dict[str, Any]]]:
    if not request.records:
        raise HTTPException(status_code=422, detail="records must not be empty.")
    results = predict_frame(prepare_raw_frame(request.records))
    record_prediction_event("/predict/batch", len(request.records), results)
    return {"predictions": results}


@app.post("/predict/processed")
def predict_processed(record: ProcessedRecord) -> dict[str, Any]:
    result = predict_frame(prepare_processed_frame([record]))[0]
    record_prediction_event("/predict/processed", 1, [result])
    return result


@app.post("/predict/processed/batch")
def predict_processed_batch(request: BatchProcessedRequest) -> dict[str, list[dict[str, Any]]]:
    if not request.records:
        raise HTTPException(status_code=422, detail="records must not be empty.")
    results = predict_frame(prepare_processed_frame(request.records))
    record_prediction_event("/predict/processed/batch", len(request.records), results)
    return {"predictions": results}
