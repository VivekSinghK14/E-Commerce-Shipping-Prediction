"""Export the deployed champion bundle for the static (in-browser) web app.

Writes deployment/static_space/model.json containing the exact XGBoost trees
(float32 split values), category maps, scaler parameters, decision threshold,
and the evaluation/override facts shown on the page. The page reproduces the
API's predictions without a server; tests/test_browser_model.py checks parity.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops_pipeline.config import FEATURE_COLUMNS, NUMERIC_COLUMNS

ARTIFACTS = PROJECT_ROOT / "artifacts"
OUTPUT = PROJECT_ROOT / "deployment" / "static_space" / "model.json"


def _float32(value: float) -> float:
    # Shortest decimal that round-trips to the same float32 value.
    return float(np.format_float_positional(np.float32(value), unique=True, trim="-"))


def export_trees(model) -> tuple[float, list]:
    raw = json.loads(model.get_booster().save_raw("json"))
    learner = raw["learner"]
    if learner["objective"]["name"] != "binary:logistic":
        raise ValueError("Only binary:logistic XGBoost models are supported")
    base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))
    trees = []
    for tree in learner["gradient_booster"]["model"]["trees"]:
        if any(tree["split_type"]):
            raise ValueError("Categorical splits are not supported")
        trees.append(
            [
                tree["left_children"],
                tree["right_children"],
                tree["split_indices"],
                [_float32(value) for value in tree["split_conditions"]],
            ]
        )
    return base_score, trees


def write_comparison_chart(evaluation: dict, path: Path) -> None:
    """Candidate vs naive baseline on the untouched test set, for the web page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("accuracy", "Accuracy"),
        ("f1_weighted", "Weighted F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("minority_recall", "Delayed recall"),
        ("roc_auc", "ROC-AUC"),
    ]
    baseline = [evaluation["naive_baseline_test_metrics"][key] for key, _ in specs]
    candidate = [evaluation["candidate_test_metrics"][key] for key, _ in specs]
    positions = np.arange(len(specs))
    figure, axis = plt.subplots(figsize=(11, 5.2))
    bars_b = axis.bar(positions - 0.18, baseline, 0.36, label="Naive majority baseline", color="#9E9E99")
    bars_c = axis.bar(positions + 0.18, candidate, 0.36, label="XGBoost model", color="#2E75B6")
    axis.bar_label(bars_b, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(bars_c, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(positions, [label for _, label in specs])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("XGBoost model vs naive baseline (untouched test set, 9,514 orders)")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    registry = ARTIFACTS / "model_registry"
    reports = ARTIFACTS / "reports"
    champion = json.loads((registry / "champion_manifest.json").read_text(encoding="utf-8"))
    bundle = joblib.load(registry / champion["bundle_file"])
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Bundle feature order does not match the pipeline contract")
    if type(bundle["model"]).__name__ != "XGBClassifier":
        raise ValueError("The browser app currently supports the XGBoost champion only")

    evaluation = json.loads((reports / "evaluation_report.json").read_text(encoding="utf-8"))
    preparation = json.loads((reports / "data_preparation_report.json").read_text(encoding="utf-8"))
    base_score, trees = export_trees(bundle["model"])
    payload = {
        "model_name": bundle["model_name"],
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "category_maps": bundle["category_maps"],
        "scaler_mean": [float(v) for v in bundle["scaler"].mean_],
        "scaler_scale": [float(v) for v in bundle["scaler"].scale_],
        "threshold": float(bundle["threshold"]),
        "base_score": base_score,
        "trees": trees,
        "facts": {
            "scope_filter": preparation["scope_filter"],
            "split_rows": preparation["split_rows"],
            "evaluation_decision": evaluation["checkpoint_decision"],
            "test_metrics": evaluation["candidate_test_metrics"],
            "baseline_test_metrics": evaluation["naive_baseline_test_metrics"],
            "business_goal_checks": evaluation["business_goal_checks"],
            "candidate_net_benefit_eur": evaluation["candidate_business_economics"]["net_benefit_eur"],
            "baseline_net_benefit_eur": evaluation["naive_baseline_business_economics"]["net_benefit_eur"],
            "override": champion.get("override"),
            "deployment_basis": (
                "DEPLOYED_BY_MANAGEMENT_OVERRIDE"
                if champion.get("promotion_status") == "approved_by_override"
                else "APPROVED_CHAMPION"
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    write_comparison_chart(evaluation, OUTPUT.parent / "evaluation.png")
    print(f"{OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KiB, {len(trees)} trees)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
