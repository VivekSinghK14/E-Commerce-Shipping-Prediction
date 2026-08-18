import argparse
import sys

import requests


SAMPLE_SHIPMENT = {
    "Customer_Region": "West",
    "Product_Category": "Home",
    "Shipping_Mode": "Standard",
    "Shipping_Cost": 122,
    "Delivery_Days": 10,
    "Order_Date": "2024-05-12",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the deployed shipping API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    if health.json().get("status") != "ok":
        raise RuntimeError(f"Health check failed: {health.text}")

    prediction = requests.post(f"{base_url}/predict", json=SAMPLE_SHIPMENT, timeout=10)
    prediction.raise_for_status()
    prediction_body = prediction.json()
    if prediction_body.get("prediction") not in [0, 1]:
        raise RuntimeError(f"Unexpected prediction response: {prediction_body}")

    monitoring = requests.get(f"{base_url}/monitoring", timeout=10)
    monitoring.raise_for_status()

    print("Deployment smoke test passed")
    print(f"Health: {health.json()}")
    print(f"Prediction: {prediction_body}")
    print(f"Monitoring: {monitoring.json()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
