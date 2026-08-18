from fastapi.testclient import TestClient

from deployment.app import app


client = TestClient(app)


RAW_SHIPMENT = {
    "Customer_Region": "West",
    "Product_Category": "Home",
    "Shipping_Mode": "Standard",
    "Shipping_Cost": 122,
    "Delivery_Days": 10,
    "Order_Date": "2024-05-12",
}


PROCESSED_SHIPMENT = {
    "Customer_Region": 1,
    "Product_Category": 2,
    "Shipping_Mode": 2,
    "Shipping_Cost": -0.4373142122501225,
    "Delivery_Days": -0.7731998756504022,
    "Order_Month": -0.4350701773353223,
    "Order_DayOfWeek": -1.003120820390247,
    "Cost_per_Day": -0.12148858585998686,
}


def test_health_reports_artifacts_and_threshold():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["threshold"] == 0.34
    assert "model_path" in body
    assert "monitoring_log_path" in body
    assert body["deployment_approval"]["decision"] in {
        "PROCEED_TO_DEPLOYMENT",
        "RETURN_TO_BUSINESS_UNDERSTANDING",
        "NO_EVALUATION_CHECKPOINT",
    }


def test_predict_raw_shipment():
    response = client.post("/predict", json=RAW_SHIPMENT)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["prediction_label"] in ["Delivered / On Time", "Not On Time"]
    assert 0 <= body["on_time_probability"] <= 1
    assert body["threshold"] == 0.34


def test_predict_processed_shipment():
    response = client.post("/predict/processed", json=PROCESSED_SHIPMENT)

    assert response.status_code == 200
    assert response.json()["prediction_label"] in ["Delivered / On Time", "Not On Time"]


def test_batch_prediction_and_monitoring():
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


def test_invalid_category_is_rejected_and_counted():
    response = client.post(
        "/predict",
        json={**RAW_SHIPMENT, "Shipping_Mode": "Teleport"},
    )

    assert response.status_code == 422
    assert "Shipping_Mode" in response.json()["detail"]
    assert client.get("/monitoring").json()["error_count"] >= 1


def test_pipeline_status_and_model_info():
    pipeline = client.get("/pipeline/status")
    model_info = client.get("/model-info")

    assert pipeline.status_code == 200
    assert model_info.status_code == 200
    assert pipeline.json()["components"]["serving_component"]["framework"] == "FastAPI"
    assert model_info.json()["metadata"]["model_type"] == "XGBClassifier"


def test_prometheus_metrics_are_exposed():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "ecommerce_prediction_requests_total" in response.text
    assert "ecommerce_predictions_total" in response.text
