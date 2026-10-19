import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


SAMPLE_SHIPMENT = {
    "Customer_Region": "West",
    "Product_Category": "Home",
    "Shipping_Mode": "Standard",
    "Shipping_Cost": 122,
    "Order_Date": "2024-05-12",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_smoke_report(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "deployment_smoke_test.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    lifecycle_path = REPORTS_DIR / "lifecycle_status.json"
    if lifecycle_path.exists():
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        lifecycle["deployment"]["deployment_validation_executed"] = True
        lifecycle["deployment"]["deployment_test_status"] = "PASSED"
        lifecycle["deployment"]["deployment_executed"] = bool(report["model_approved"])
        lifecycle["deployment"]["smoke_test_report"] = str(
            REPORTS_DIR / "deployment_smoke_test.json"
        )
        lifecycle_path.write_text(
            json.dumps(lifecycle, indent=2, sort_keys=True), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the deployed shipping API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="Fail unless an approved model is actively serving predictions.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    if health.json().get("status") != "ok":
        raise RuntimeError(f"Health check failed: {health.text}")

    health_body = health.json()
    approved = bool(health_body.get("deployment_approval", {}).get("approved"))
    if args.require_approved and not approved:
        raise RuntimeError("No approved model is available for production smoke testing")

    prediction = requests.post(f"{base_url}/predict", json=SAMPLE_SHIPMENT, timeout=10)
    if approved:
        prediction.raise_for_status()
        prediction_body = prediction.json()
        if prediction_body.get("prediction") not in [0, 1]:
            raise RuntimeError(f"Unexpected prediction response: {prediction_body}")
        gate_behavior = "approved prediction served"
    else:
        if prediction.status_code != 503:
            raise RuntimeError(
                f"Rejected model must be blocked with HTTP 503, got {prediction.status_code}"
            )
        prediction_body = prediction.json()
        gate_behavior = "unapproved prediction correctly blocked"

    pipeline = requests.get(f"{base_url}/pipeline/status", timeout=10)
    pipeline.raise_for_status()
    monitoring = requests.get(f"{base_url}/monitoring", timeout=10)
    monitoring.raise_for_status()
    metrics = requests.get(f"{base_url}/metrics", timeout=10)
    metrics.raise_for_status()
    if "ecommerce_prediction_requests_total" not in metrics.text:
        raise RuntimeError("Prometheus metrics endpoint is incomplete")

    report = {
        "generated_at": utc_now(),
        "status": "passed",
        "base_url": base_url,
        "model_approved": approved,
        "gate_behavior": gate_behavior,
        "health_status": health_body.get("status"),
        "pipeline_status": pipeline.json().get("status"),
        "prediction_http_status": prediction.status_code,
        "monitoring_available": True,
        "metrics_available": True,
    }
    write_smoke_report(report)

    print("Deployment smoke test passed")
    print(f"Mode: {gate_behavior}")
    print(f"Health: {health_body}")
    print(f"Prediction: {prediction_body}")
    print(f"Pipeline: {pipeline.json()}")
    print(f"Monitoring: {monitoring.json()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
