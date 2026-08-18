import json
import os
import pickle
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "processed" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "processed" / "scaler.pkl"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "processed" / "model_comparison_all.csv"
MLFLOW_EXPORT_PATH = PROJECT_ROOT / "processed" / "mlflow_runs_export.csv"
THRESHOLD_SWEEP_PATH = PROJECT_ROOT / "processed" / "threshold_sweep.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "reports" / "checkpoint_decision.json"
REGISTRY_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "model_registry" / "candidate_manifest.json"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", str(RUNTIME_DIR / "prediction_log.jsonl"))
)

FEATURE_COLUMNS = [
    "Customer_Region",
    "Product_Category",
    "Shipping_Mode",
    "Shipping_Cost",
    "Delivery_Days",
    "Order_Month",
    "Order_DayOfWeek",
    "Cost_per_Day",
]

NUMERIC_COLUMNS = [
    "Shipping_Cost",
    "Delivery_Days",
    "Cost_per_Day",
    "Order_Month",
    "Order_DayOfWeek",
]

SCALER_MEAN = {
    "Shipping_Cost": 138.772625,
    "Delivery_Days": 5.99135,
    "Cost_per_Day": 29.75314143849206,
    "Order_Month": 6.504175,
    "Order_DayOfWeek": 3.007375,
}

SCALER_SCALE = {
    "Shipping_Cost": 72.65399593697083,
    "Delivery_Days": 2.5754660117151613,
    "Cost_per_Day": 24.719535726204956,
    "Order_Month": 3.4573158041137924,
    "Order_DayOfWeek": 2.0011298332129774,
}

CATEGORY_MAPS = {
    "Customer_Region": {
        "Central": 0,
        "East": 1,
        "North": 2,
        "South": 3,
        "West": 4,
    },
    "Product_Category": {
        "Beauty": 0,
        "Electronics": 1,
        "Fashion": 2,
        "Grocery": 3,
        "Home": 4,
        "Sports": 5,
    },
    "Shipping_Mode": {
        "Express": 0,
        "Same Day": 1,
        "Standard": 2,
    },
}

DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.34"))
TARGET_LABELS = {
    0: "Not On Time",
    1: "Delivered / On Time",
}

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
    Delivery_Days: float = Field(..., examples=[10.0])
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
    Cost_per_Day: float | None = None


class ProcessedRecord(BaseModel):
    Customer_Region: int
    Product_Category: int
    Shipping_Mode: int
    Shipping_Cost: float
    Delivery_Days: float
    Order_Month: float
    Order_DayOfWeek: float
    Cost_per_Day: float


class BatchShipmentRequest(BaseModel):
    records: list[ShipmentRecord]


class BatchProcessedRequest(BaseModel):
    records: list[ProcessedRecord]


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Required artifact not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)

app = FastAPI(
    title="E-Commerce Shipping Prediction API",
    version="1.0.0",
    description="Predicts whether an e-commerce shipment will be delivered/on time.",
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
        "threshold": DEFAULT_THRESHOLD,
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


def monitoring_snapshot() -> dict[str, Any]:
    with MONITORING_LOCK:
        probability_count = MONITORING_STATE["probability_count"]
        average_probability = (
            MONITORING_STATE["probability_sum"] / probability_count
            if probability_count
            else None
        )
        return {
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


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def deployment_approval() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {
            "approved": False,
            "decision": "NO_EVALUATION_CHECKPOINT",
            "serving_mode": "legacy_prototype_only",
        }
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    decision = checkpoint.get("checkpoint_decision")
    return {
        "approved": decision == "PROCEED_TO_DEPLOYMENT",
        "decision": decision,
        "serving_mode": (
            "approved_candidate" if decision == "PROCEED_TO_DEPLOYMENT" else "legacy_prototype_only"
        ),
        "warning": (
            None
            if decision == "PROCEED_TO_DEPLOYMENT"
            else "Current endpoint is for technical demonstration; the evaluated candidate is not approved for business use."
        ),
    }


def selected_model_metadata() -> dict[str, Any]:
    rows = read_csv_records(MODEL_COMPARISON_PATH)
    selected = next(
        (row for row in rows if str(row.get("Model", "")).casefold() == "xgboost binary"),
        None,
    )
    threshold_rows = read_csv_records(THRESHOLD_SWEEP_PATH)
    best_threshold = None
    if threshold_rows:
        best_threshold = max(threshold_rows, key=lambda row: row.get("F1", 0))
    return {
        "model_type": type(model).__name__,
        "selected_model": selected,
        "best_threshold_from_sweep": best_threshold,
        "configured_threshold": DEFAULT_THRESHOLD,
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
    mapping = CATEGORY_MAPS[name]

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


def raw_record_to_features(record: ShipmentRecord) -> dict[str, float | int]:
    if record.Order_Month is None or record.Order_DayOfWeek is None:
        if not record.Order_Date:
            raise HTTPException(
                status_code=422,
                detail="Provide Order_Date or both Order_Month and Order_DayOfWeek.",
            )
        parsed_date = parse_order_date(record.Order_Date)
        order_month = parsed_date.month
        order_day_of_week = parsed_date.weekday()
    else:
        order_month = record.Order_Month
        order_day_of_week = record.Order_DayOfWeek

    if record.Delivery_Days <= 0:
        raise HTTPException(status_code=422, detail="Delivery_Days must be greater than 0.")

    cost_per_day = record.Cost_per_Day
    if cost_per_day is None:
        cost_per_day = record.Shipping_Cost / record.Delivery_Days

    return {
        "Customer_Region": canonicalize_category("Customer_Region", record.Customer_Region),
        "Product_Category": canonicalize_category("Product_Category", record.Product_Category),
        "Shipping_Mode": canonicalize_category("Shipping_Mode", record.Shipping_Mode),
        "Shipping_Cost": record.Shipping_Cost,
        "Delivery_Days": record.Delivery_Days,
        "Order_Month": order_month,
        "Order_DayOfWeek": order_day_of_week,
        "Cost_per_Day": cost_per_day,
    }


def prepare_raw_frame(records: list[ShipmentRecord]) -> pd.DataFrame:
    rows = [raw_record_to_features(record) for record in records]
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    frame[NUMERIC_COLUMNS] = scale_numeric_columns(frame[NUMERIC_COLUMNS])
    return frame


def scale_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if hasattr(scaler, "transform"):
        return pd.DataFrame(
            scaler.transform(frame),
            columns=NUMERIC_COLUMNS,
            index=frame.index,
        )

    means = pd.Series(SCALER_MEAN)
    scales = pd.Series(SCALER_SCALE)
    return (frame - means) / scales


def prepare_processed_frame(records: list[ProcessedRecord]) -> pd.DataFrame:
    rows = [record.model_dump() for record in records]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def predict_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if list(frame.columns) != FEATURE_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"Expected columns in this order: {FEATURE_COLUMNS}",
        )

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        probabilities = proba[:, positive_index]
        predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    else:
        predictions = model.predict(frame).astype(int)

    results = []
    for idx, prediction in enumerate(predictions):
        probability = None if probabilities is None else float(probabilities[idx])
        prediction_int = int(prediction)
        results.append(
            {
                "prediction": prediction_int,
                "prediction_label": TARGET_LABELS.get(prediction_int, str(prediction_int)),
                "on_time_probability": probability,
                "threshold": DEFAULT_THRESHOLD if probabilities is not None else None,
            }
        )
    return results


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "scaler_source": "artifact" if hasattr(scaler, "transform") else "embedded_parameters",
        "monitoring_log_path": str(PREDICTION_LOG_PATH),
        "features": FEATURE_COLUMNS,
        "threshold": DEFAULT_THRESHOLD,
        "deployment_approval": deployment_approval(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "target": TARGET_LABELS,
        "features": FEATURE_COLUMNS,
        "artifacts": {
            "model": artifact_status(MODEL_PATH),
            "scaler": artifact_status(SCALER_PATH),
            "model_comparison": artifact_status(MODEL_COMPARISON_PATH),
            "mlflow_export": artifact_status(MLFLOW_EXPORT_PATH),
            "threshold_sweep": artifact_status(THRESHOLD_SWEEP_PATH),
        },
        "metadata": selected_model_metadata(),
        "deployment_approval": deployment_approval(),
    }


@app.get("/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    feature_store_artifacts = {
        "X_train": artifact_status(PROJECT_ROOT / "processed" / "X_train.csv"),
        "X_test": artifact_status(PROJECT_ROOT / "processed" / "X_test.csv"),
        "y_train_binary": artifact_status(PROJECT_ROOT / "processed" / "y_train_binary.csv"),
        "y_test_binary": artifact_status(PROJECT_ROOT / "processed" / "y_test_binary.csv"),
    }
    return {
        "status": "ready",
        "deployment_approval": deployment_approval(),
        "components": {
            "source_code_repository": "current Git repository",
            "feature_store": feature_store_artifacts,
            "model_registry": {"production_model": artifact_status(MODEL_PATH)},
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
                "promotion_requires_manual_flag": True,
            },
        },
    }


@app.get("/monitoring")
def monitoring() -> dict[str, Any]:
    return monitoring_snapshot()


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
