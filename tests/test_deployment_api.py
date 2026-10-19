import importlib
import json
from collections import Counter

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


deployment_app = importlib.import_module("deployment.app")


RAW_SHIPMENT = {
    "Customer_Region": "West",
    "Product_Category": "Home",
    "Shipping_Mode": "Standard",
    "Shipping_Cost": 122,
    "Order_Date": "2024-05-12",
}


PROCESSED_SHIPMENT = {
    "Customer_Region": 1,
    "Product_Category": 2,
    "Shipping_Mode": 2,
    "Shipping_Cost": 0.0,
    "Order_Month": 0.0,
    "Order_DayOfWeek": 0.0,
}


def configure_test_bundle(tmp_path, monkeypatch, approved=True):
    module = deployment_app
    artifact_root = tmp_path / "artifacts"
    registry = artifact_root / "model_registry"
    feature_store = artifact_root / "feature_store"
    reports = artifact_root / "reports"
    registry.mkdir(parents=True)
    feature_store.mkdir(parents=True)
    reports.mkdir(parents=True)

    category_maps = {
        "Customer_Region": {"Central": 0, "East": 1, "North": 2, "South": 3, "West": 4},
        "Product_Category": {"Beauty": 0, "Electronics": 1, "Fashion": 2, "Grocery": 3, "Home": 4, "Sports": 5},
        "Shipping_Mode": {"Express": 0, "Same Day": 1, "Standard": 2},
    }
    feature_columns = [
        "Customer_Region",
        "Product_Category",
        "Shipping_Mode",
        "Shipping_Cost",
        "Order_Month",
        "Order_DayOfWeek",
    ]
    numeric_columns = ["Shipping_Cost", "Order_Month", "Order_DayOfWeek"]
    sample = pd.DataFrame(
        [[0, 0, 0, 10.0, 1.0, 0.0], [1, 1, 1, 20.0, 2.0, 1.0]],
        columns=feature_columns,
    )
    model = DummyClassifier(strategy="constant", constant=1).fit(sample, [1, 1])
    bundle = {
        "model": model,
        "scaler": StandardScaler().fit(sample[numeric_columns]),
        "category_maps": category_maps,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "threshold": 0.5,
        "model_name": "test_dummy",
    }
    bundle_path = registry / "candidate_bundle.joblib"
    model_path = registry / "candidate_model.joblib"
    joblib.dump(bundle, bundle_path)
    joblib.dump(model, model_path)
    (reports / "checkpoint_decision.json").write_text(
        json.dumps(
            {"checkpoint_decision": "PROCEED_TO_DEPLOYMENT" if approved else "RETURN_TO_BUSINESS_UNDERSTANDING"}
        ),
        encoding="utf-8",
    )
    (reports / "analytical_model_selection.json").write_text(
        json.dumps(
            {
                "selected_model": "test_dummy",
                "validation_selected_threshold": 0.5,
                "validation_metrics_at_default_threshold": {"roc_auc": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (registry / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "selected_model": "test_dummy",
                "bundle_file": "candidate_bundle.joblib",
                "validation_selected_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    for name, value in {
        "ARTIFACTS_DIR": artifact_root,
        "REPORTS_DIR": reports,
        "REGISTRY_DIR": registry,
        "FEATURE_STORE_DIR": feature_store,
        "MODEL_PATH": model_path,
        "SCALER_PATH": feature_store / "scaler.joblib",
        "MODEL_COMPARISON_PATH": reports / "model_comparison_validation.csv",
        "MLFLOW_EXPORT_PATH": artifact_root / "mlflow",
        "THRESHOLD_SWEEP_PATH": reports / "threshold_sweep_validation.csv",
        "BASELINE_COMPARISON_PATH": reports / "naive_vs_best_candidate.png",
        "SELECTION_PATH": reports / "analytical_model_selection.json",
        "CHECKPOINT_PATH": reports / "checkpoint_decision.json",
        "REGISTRY_MANIFEST_PATH": registry / "candidate_manifest.json",
        "CHAMPION_MANIFEST_PATH": registry / "champion_manifest.json",
        "SYSTEM_STATE_PATH": artifact_root / "system_state.json",
        "PREDICTION_LOG_PATH": tmp_path / "prediction_log.jsonl",
    }.items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(deployment_app.operations_store, "DB_PATH", tmp_path / "operations.sqlite3")
    module._load_bundle.cache_clear()
    module.MONITORING_STATE.update(
        {
            "request_count": 0,
            "prediction_count": 0,
            "batch_request_count": 0,
            "error_count": 0,
            "class_counts": Counter(),
            "probability_sum": 0.0,
            "probability_count": 0,
            "last_prediction_at": None,
        }
    )
    module.RECENT_PREDICTIONS.clear()
    return TestClient(module.app)


def test_health_reports_approved_registry_bundle(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["threshold"] == 0.5
    assert body["model_ready"] is True
    assert "model_path" in body
    assert "monitoring_log_path" in body
    assert body["deployment_approval"]["decision"] in {
        "PROCEED_TO_DEPLOYMENT",
        "RETURN_TO_BUSINESS_UNDERSTANDING",
        "NO_EVALUATION_CHECKPOINT",
    }


def test_rejected_checkpoint_blocks_predictions(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch, approved=False)
    response = client.post("/predict", json=RAW_SHIPMENT)

    assert response.status_code == 503
    assert "blocked" in response.json()["detail"]


def test_predict_raw_shipment_uses_order_time_fields(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    response = client.post("/predict", json=RAW_SHIPMENT)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["prediction_id"]
    assert body["prediction_label"] in ["Delivered", "Delayed"]
    assert 0 <= body["on_time_probability"] <= 1
    assert body["threshold"] == 0.5


def test_predict_processed_shipment(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    response = client.post("/predict/processed", json=PROCESSED_SHIPMENT)

    assert response.status_code == 200
    assert response.json()["prediction_label"] in ["Delivered", "Delayed"]


def test_batch_prediction_and_monitoring(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    before = client.get("/monitoring").json()["prediction_count"]

    response = client.post(
        "/predict/batch",
        json={"records": [RAW_SHIPMENT, {**RAW_SHIPMENT, "Shipping_Mode": "Express"}]},
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2

    after = client.get("/monitoring").json()
    assert after["prediction_count"] >= before + 2
    assert after["request_count"] >= 1
    assert after["class_counts"]


def test_invalid_category_is_rejected_and_counted(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    response = client.post(
        "/predict",
        json={**RAW_SHIPMENT, "Shipping_Mode": "Teleport"},
    )

    assert response.status_code == 422
    assert "Shipping_Mode" in response.json()["detail"]
    assert client.get("/monitoring").json()["error_count"] >= 1


def test_pipeline_status_and_model_info(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    pipeline = client.get("/pipeline/status")
    model_info = client.get("/model-info")

    assert pipeline.status_code == 200
    assert model_info.status_code == 200
    assert pipeline.json()["components"]["serving_component"]["framework"] == "FastAPI"
    assert model_info.json()["metadata"]["model_type"] == "test_dummy"
    assert pipeline.json()["status"] == "ready"


def test_prometheus_metrics_are_exposed(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "ecommerce_prediction_requests_total" in response.text
    assert "ecommerce_predictions_total" in response.text


def test_outcome_feedback_records_observed_value(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    prediction = client.post("/predict", json=RAW_SHIPMENT).json()
    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction["prediction_id"],
            "actual_delivered": False,
            "realized_value_eur": 18.5,
        },
    )

    assert response.status_code == 200
    assert response.json()["outcome_feedback_count"] == 1
    assert response.json()["observed_delayed_recall"] == 0.0
    assert response.json()["observed_performance"]["sample_size"] == 1
    assert response.json()["observed_performance"]["retraining_recommended"] is False
    assert response.json()["realized_value_eur"] == 18.5


def test_support_ticket_can_be_opened_and_resolved(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    created = client.post(
        "/support",
        json={"category": "prediction", "description": "Review unexpected result"},
    )
    ticket_id = created.json()["ticket_id"]

    assert created.status_code == 201
    assert created.json()["status"] == "open"
    resolved = client.post(
        f"/support/{ticket_id}/resolve",
        json={"resolution": "Confirmed expected behavior"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"
    assert client.get("/utilization").json()["open_support_tickets"] == 0


def test_override_champion_serves_predictions_with_warning(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch, approved=False)
    registry = tmp_path / "artifacts" / "model_registry"
    (registry / "champion_bundle.joblib").write_bytes(
        (registry / "candidate_bundle.joblib").read_bytes()
    )
    (registry / "champion_manifest.json").write_text(
        json.dumps(
            {
                "selected_model": "test_dummy",
                "bundle_file": "champion_bundle.joblib",
                "threshold": 0.5,
                "promotion_status": "approved_by_override",
                "override": {"actor": "tester", "rationale": "course demo"},
            }
        ),
        encoding="utf-8",
    )

    health = client.get("/health").json()
    response = client.post("/predict", json=RAW_SHIPMENT)

    assert health["model_ready"] is True
    assert health["deployment_approval"]["decision"] == "DEPLOYED_BY_MANAGEMENT_OVERRIDE"
    assert health["deployment_approval"]["evaluation_decision"] == "RETURN_TO_BUSINESS_UNDERSTANDING"
    assert response.status_code == 200
    assert response.json()["prediction_label"] in ["Delivered", "Delayed"]
    assert response.json()["deployment_basis"] == "DEPLOYED_BY_MANAGEMENT_OVERRIDE"
    assert "did not meet all evaluation gates" in response.json()["warning"]


def test_web_app_is_served_at_root(tmp_path, monkeypatch):
    client = configure_test_bundle(tmp_path, monkeypatch)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Shipping Delay Predictor" in response.text
