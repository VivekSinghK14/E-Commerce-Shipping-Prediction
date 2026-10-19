import json
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.config import NUMERIC_COLUMNS, PipelineConfig
from mlops_pipeline.feature_engineering import build_order_time_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_SPACE = PROJECT_ROOT / "deployment" / "static_space"
REGISTRY = PROJECT_ROOT / "artifacts" / "model_registry"

NODE_RUNNER = """
const fs = require("fs");
const predictor = require(process.argv[2]);
const model = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const records = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const out = records.map((r) => predictor.predict(model, r).on_time_probability);
fs.writeFileSync(process.argv[5], JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the browser parity test")
@pytest.mark.skipif(
    not (STATIC_SPACE / "model.json").exists() or not (REGISTRY / "champion_manifest.json").exists(),
    reason="Run scripts/export_browser_model.py after a deployment override",
)
def test_browser_predictor_matches_the_served_python_model(tmp_path):
    champion = json.loads((REGISTRY / "champion_manifest.json").read_text(encoding="utf-8"))
    bundle = joblib.load(REGISTRY / champion["bundle_file"])

    raw = pd.read_csv(PipelineConfig().raw_data)
    raw = raw[raw["Delivery_Status"].isin(["Delivered", "Delayed"])].reset_index(drop=True)
    raw["Order_Date"] = pd.to_datetime(raw["Order_Date"], format="mixed").dt.strftime("%Y-%m-%d")
    records = raw[
        ["Customer_Region", "Product_Category", "Shipping_Mode", "Shipping_Cost", "Order_Date"]
    ]

    # Python reference: the same path as deployment/app.py prepare_raw_frame + predict_frame.
    features = build_order_time_features(records, bundle["category_maps"])
    features[NUMERIC_COLUMNS] = bundle["scaler"].transform(features[NUMERIC_COLUMNS])
    expected = bundle["model"].predict_proba(features)[:, 1]

    paths = {name: tmp_path / f"{name}.json" for name in ["records", "out"]}
    paths["records"].write_text(records.to_json(orient="records"), encoding="utf-8")
    runner = tmp_path / "runner.js"
    runner.write_text(NODE_RUNNER, encoding="utf-8")
    subprocess.run(
        ["node", str(runner), str(STATIC_SPACE / "predictor.js"), str(STATIC_SPACE / "model.json"),
         str(paths["records"]), str(paths["out"])],
        check=True,
    )
    actual = np.array(json.loads(paths["out"].read_text(encoding="utf-8")))

    threshold = float(bundle["threshold"])
    assert len(actual) == len(expected) == len(raw)
    assert np.max(np.abs(actual - expected)) < 1e-5
    near_threshold = np.abs(expected - threshold) < 1e-5
    assert np.array_equal((actual >= threshold)[~near_threshold], (expected >= threshold)[~near_threshold])
