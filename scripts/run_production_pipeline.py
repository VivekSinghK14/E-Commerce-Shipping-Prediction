import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
DEFAULT_THRESHOLD = 0.34
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


def load_target(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    return frame.iloc[:, 0]


def validate_artifacts() -> dict[str, str]:
    required = {
        "X_train_balanced": PROCESSED_DIR / "X_train_balanced.csv",
        "y_train_binary": PROCESSED_DIR / "y_train_binary.csv",
        "X_test": PROCESSED_DIR / "X_test.csv",
        "y_test_binary": PROCESSED_DIR / "y_test_binary.csv",
        "production_model": PROCESSED_DIR / "best_model.pkl",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")
    return {name: str(path) for name, path in required.items()}


def train_candidate(quick: bool = False) -> XGBClassifier:
    x_train = pd.read_csv(PROCESSED_DIR / "X_train_balanced.csv")[FEATURE_COLUMNS]
    y_train = load_target(PROCESSED_DIR / "y_train_binary.csv")

    if len(x_train) != len(y_train):
        raise ValueError(
            "X_train_balanced.csv and y_train_binary.csv must contain the same number of rows."
        )

    if quick:
        sample_size = min(5000, len(x_train))
        x_train = x_train.iloc[:sample_size]
        y_train = y_train.iloc[:sample_size]

    model = XGBClassifier(
        n_estimators=20 if quick else 200,
        max_depth=4 if quick else 6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    return model


def evaluate(model: XGBClassifier) -> dict[str, float]:
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")[FEATURE_COLUMNS]
    y_test = load_target(PROCESSED_DIR / "y_test_binary.csv")
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "precision_weighted": float(precision_score(y_test, predictions, average="weighted")),
        "recall_weighted": float(recall_score(y_test, predictions, average="weighted")),
        "threshold": DEFAULT_THRESHOLD,
    }


def write_artifacts(
    model: XGBClassifier,
    metrics: dict[str, float],
    input_artifacts: dict[str, str],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = output_dir / f"candidate_model_{timestamp}.pkl"
    metadata_path = output_dir / f"pipeline_run_{timestamp}.json"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "timestamp": timestamp,
        "pipeline": "ecommerce_shipping_binary_xgboost",
        "model_type": type(model).__name__,
        "input_artifacts": input_artifacts,
        "candidate_model": str(model_path),
        "metrics": metrics,
        "promotion_policy": "manual --promote flag required",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return model_path, metadata_path


def promote_model(candidate_model_path: Path, backup_dir: Path) -> Path:
    production_model = PROCESSED_DIR / "best_model.pkl"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"best_model_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.pkl"
    shutil.copy2(production_model, backup_path)
    shutil.copy2(candidate_model_path, production_model)
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the prototype ML production pipeline.")
    parser.add_argument("--quick", action="store_true", help="Train a smaller candidate for smoke tests.")
    parser.add_argument("--promote", action="store_true", help="Promote candidate model to processed/best_model.pkl.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runtime" / "pipeline_runs"),
        help="Directory for candidate model and metadata.",
    )
    args = parser.parse_args()

    input_artifacts = validate_artifacts()
    model = train_candidate(quick=args.quick)
    metrics = evaluate(model)
    candidate_model_path, metadata_path = write_artifacts(
        model=model,
        metrics=metrics,
        input_artifacts=input_artifacts,
        output_dir=Path(args.output_dir),
    )

    print(f"Candidate model: {candidate_model_path}")
    print(f"Pipeline metadata: {metadata_path}")
    print(f"Metrics: {json.dumps(metrics, sort_keys=True)}")

    if args.promote:
        raise RuntimeError(
            "Direct promotion is disabled because it bypasses the BPMN evaluation and business gates. "
            "Use `python scripts/retrain_model.py --force`; it promotes only an approved candidate."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
