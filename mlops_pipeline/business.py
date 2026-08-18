from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix

from .config import PipelineConfig


def classification_economics(
    y_true: Iterable[int], prediction: Iterable[int], config: PipelineConfig
) -> dict:
    """Estimate business value for Delivered=1 and Delayed/Returned=0.

    These are scenario assumptions, not accounting facts. A correct risk flag
    (true negative in the Delivered-positive convention) enables intervention;
    a missed risk (false positive) is the most expensive error.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    components = {
        "correct_risk_flags": {
            "count": int(tn),
            "unit_value_eur": config.benefit_correct_risk_flag,
            "total_eur": float(tn * config.benefit_correct_risk_flag),
        },
        "correct_on_time_predictions": {
            "count": int(tp),
            "unit_value_eur": config.benefit_correct_on_time,
            "total_eur": float(tp * config.benefit_correct_on_time),
        },
        "missed_risks": {
            "count": int(fp),
            "unit_cost_eur": config.cost_missed_risk,
            "total_eur": float(-fp * config.cost_missed_risk),
        },
        "unnecessary_interventions": {
            "count": int(fn),
            "unit_cost_eur": config.cost_unnecessary_intervention,
            "total_eur": float(-fn * config.cost_unnecessary_intervention),
        },
    }
    net = sum(item["total_eur"] for item in components.values())
    return {
        "currency": "EUR",
        "positive_class": "Delivered / On Time",
        "negative_class": "Delayed or Returned / At Risk",
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "components": components,
        "net_benefit_eur": float(net),
        "net_benefit_per_1000_predictions_eur": float(net / len(np.asarray(y_true)) * 1000),
        "assumption_warning": (
            "Illustrative scenario values; validate unit benefits/costs with finance and operations."
        ),
    }


def model_business_characteristics(name: str) -> dict:
    characteristics = {
        "dummy_majority": ("high", "very low", "benchmark only"),
        "logistic_regression": ("high", "low", "coefficients show direction and magnitude"),
        "random_forest": ("medium", "medium", "nonlinear interactions; global importances available"),
        "xgboost": ("medium", "medium", "strong nonlinear capacity; SHAP/importances recommended"),
    }
    interpretability, operational_complexity, explanation = characteristics.get(
        name, ("unknown", "unknown", "model-specific review required")
    )
    return {
        "interpretability": interpretability,
        "operational_complexity": operational_complexity,
        "business_explanation": explanation,
    }
