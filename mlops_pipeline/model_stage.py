import time
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .business import classification_economics, model_business_characteristics
from .config import FEATURE_COLUMNS, PipelineConfig
from .io_utils import file_sha256, utc_now, write_json


@dataclass
class CandidateResult:
    name: str
    model: object
    metrics: dict
    training_seconds: float
    inference_ms_per_1000: float
    economics: dict
    mlflow_run_id: str


def _load_split(config: PipelineConfig, split: str) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_csv(config.feature_store / f"X_{split}.csv")
    targets = pd.read_csv(config.feature_store / f"y_{split}.csv")["binary_target"]
    return features, targets


def _metrics(y_true: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_weighted": float(f1_score(y_true, prediction, average="weighted")),
        "precision_weighted": float(
            precision_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, prediction, average="weighted")),
        "minority_recall": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def _candidate_models(random_state: int, quick: bool) -> dict:
    models = {
        "dummy_majority": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=40 if quick else 200,
            max_depth=8 if quick else 14,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=30 if quick else 200,
            max_depth=4 if quick else 6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )
    except ImportError:
        pass
    return models


def _tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        prediction = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
                "f1_weighted": float(f1_score(y_true, prediction, average="weighted")),
                "minority_recall": float(
                    recall_score(y_true, prediction, pos_label=0, zero_division=0)
                ),
            }
        )
    # Balanced accuracy prevents the majority-only threshold from appearing optimal.
    winner = max(rows, key=lambda row: (row["balanced_accuracy"], row["f1_weighted"]))
    return winner["threshold"], rows


def develop_and_assess_models(config: PipelineConfig, quick: bool = False) -> dict:
    """Implement test design, model selection/development, and analytical assessment."""
    x_train, y_train = _load_split(config, "train")
    x_validation, y_validation = _load_split(config, "validation")
    config.metadata_store.mkdir(parents=True, exist_ok=True)
    config.model_registry.mkdir(parents=True, exist_ok=True)
    config.mlflow_store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_store.resolve().as_uri())
    mlflow.set_experiment("ecommerce-shipping-bpmn")

    hypotheses = {
        "generated_at": utc_now(),
        "business_hypothesis": (
            "Attributes available at or near order time contain enough signal to identify "
            "shipments that will be delayed or returned."
        ),
        "null_hypothesis": (
            "The available predictors provide no useful discrimination beyond chance/baseline."
        ),
        "acceptance_evidence": {
            "minimum_test_roc_auc": config.minimum_roc_auc,
            "minimum_test_minority_recall": config.minimum_minority_recall,
            "minimum_weighted_f1_uplift_over_baseline": config.minimum_f1_improvement,
        },
    }
    write_json(config.reports / "hypotheses_account.json", hypotheses)
    write_json(
        config.reports / "analytical_requirements.json",
        {
            "generated_at": utc_now(),
            "task": "supervised binary classification",
            "explainability": "feature importance available for tree candidates",
            "reproducibility": "fixed seed, portable paths, persisted split artifacts",
            "validation": "model/threshold chosen without test-set access",
            "promotion": "all business-goal gates must pass",
        },
    )

    test_design = {
        "generated_at": utc_now(),
        "problem": "Binary classification: Delivered (1) vs Delayed/Returned (0)",
        "selection_dataset": "validation split only",
        "final_assessment_dataset": "untouched test split only",
        "selection_metric": "validation ROC-AUC",
        "threshold_metric": "validation balanced accuracy",
        "baseline": "DummyClassifier predicting training class prior",
        "models": list(_candidate_models(config.random_state, quick)),
        "random_state": config.random_state,
    }
    write_json(config.reports / "analytical_model_test_design.json", test_design)

    results = []
    fitted = {}
    for name, model in _candidate_models(config.random_state, quick).items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        inference_start = time.perf_counter()
        probability = model.predict_proba(x_validation)[:, 1]
        prediction = model.predict(x_validation)
        inference_seconds = time.perf_counter() - inference_start
        inference_ms_per_1000 = inference_seconds / len(x_validation) * 1_000_000
        metrics = _metrics(y_validation, prediction, probability)
        economics = classification_economics(y_validation, prediction, config)
        business_characteristics = model_business_characteristics(name)
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    "pipeline_stage": "analysis_modeling",
                    "dataset_split": "validation",
                    "problem": "binary_shipping_outcome",
                }
            )
            mlflow.log_params(
                {
                    "model_name": name,
                    "random_state": config.random_state,
                    "training_rows": len(x_train),
                    "feature_count": len(x_train.columns),
                    "quick_mode": quick,
                }
            )
            mlflow.log_metrics(
                {
                    **metrics,
                    "training_seconds": elapsed,
                    "inference_ms_per_1000": inference_ms_per_1000,
                    "validation_net_benefit_eur": economics["net_benefit_eur"],
                }
            )
            input_example = x_train.head(5)
            signature = infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=input_example,
                signature=signature,
            )
            run_id = run.info.run_id
        result = CandidateResult(
            name, model, metrics, elapsed, inference_ms_per_1000, economics, run_id
        )
        results.append(result)
        fitted[name] = model
        write_json(
            config.metadata_store / f"{name}.json",
            {
                "generated_at": utc_now(),
                "name": name,
                "estimator": type(model).__name__,
                "parameters": model.get_params(),
                "validation_metrics": metrics,
                "training_seconds": elapsed,
                "inference_ms_per_1000": inference_ms_per_1000,
                "training_rows": len(x_train),
                "feature_columns": list(x_train.columns),
                "business_economics": economics,
                "business_characteristics": business_characteristics,
                "mlflow_run_id": run_id,
                "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            },
        )

    comparison = pd.DataFrame(
        [
            {
                "model": result.name,
                **result.metrics,
                "training_seconds": result.training_seconds,
                "inference_ms_per_1000": result.inference_ms_per_1000,
                "net_benefit_eur": result.economics["net_benefit_eur"],
                **model_business_characteristics(result.name),
                "mlflow_run_id": result.mlflow_run_id,
            }
            for result in results
        ]
    ).sort_values("roc_auc", ascending=False)
    comparison.to_csv(config.reports / "model_comparison_validation.csv", index=False)
    comparison.to_csv(config.reports / "model_business_factors.csv", index=False)

    eligible = [result for result in results if result.name != "dummy_majority"]
    selected = max(eligible, key=lambda result: result.metrics["roc_auc"])
    validation_probability = selected.model.predict_proba(x_validation)[:, 1]
    threshold, threshold_rows = _tune_threshold(y_validation, validation_probability)
    pd.DataFrame(threshold_rows).to_csv(
        config.reports / "threshold_sweep_validation.csv", index=False
    )

    # Controlled ablation: same selected algorithm and split, only Cost_per_Day differs.
    without_cost_per_day = [column for column in FEATURE_COLUMNS if column != "Cost_per_Day"]
    ablation_model = clone(selected.model)
    ablation_model.fit(x_train[without_cost_per_day], y_train)
    ablation_probability = ablation_model.predict_proba(x_validation[without_cost_per_day])[:, 1]
    ablation_prediction = ablation_model.predict(x_validation[without_cost_per_day])
    ablation_metrics = _metrics(y_validation, ablation_prediction, ablation_probability)
    importance = None
    if hasattr(selected.model, "feature_importances_"):
        importance = {
            column: float(value)
            for column, value in zip(FEATURE_COLUMNS, selected.model.feature_importances_)
        }
    elif hasattr(selected.model, "coef_"):
        importance = {
            column: float(value)
            for column, value in zip(FEATURE_COLUMNS, selected.model.coef_[0])
        }
    cost_per_day_ablation = {
        "generated_at": utc_now(),
        "model": selected.name,
        "controlled_comparison": "same algorithm, split, seed, and settings; only feature set differs",
        "with_cost_per_day": selected.metrics,
        "without_cost_per_day": ablation_metrics,
        "delta_with_minus_without": {
            metric: selected.metrics[metric] - ablation_metrics[metric]
            for metric in selected.metrics
        },
        "material_improvement_threshold_roc_auc": 0.01,
        "material_improvement": (
            selected.metrics["roc_auc"] - ablation_metrics["roc_auc"] >= 0.01
            and selected.metrics["roc_auc"] >= config.minimum_roc_auc
        ),
        "cost_per_day_formula": "Shipping_Cost / Delivery_Days",
        "feature_importance_or_coefficient": None if importance is None else importance.get("Cost_per_Day"),
        "all_feature_importances_or_coefficients": importance,
        "interpretation": (
            "A positive ROC-AUC delta means Cost_per_Day improved ranking; a zero/negative delta "
            "means it added no validated improvement. Because it is derived from two existing "
            "features, apparent importance does not by itself prove new information or causality."
        ),
    }
    cost_per_day_ablation["conclusion"] = (
        "Cost_per_Day provides a material validated improvement."
        if cost_per_day_ablation["material_improvement"]
        else "Cost_per_Day does not provide a material validated improvement; keep only if its business interpretability is useful."
    )
    write_json(config.reports / "cost_per_day_ablation.json", cost_per_day_ablation)
    ablation_md = f"""# Cost_per_Day feature ablation

`Cost_per_Day = Shipping_Cost / Delivery_Days` was tested using the same
`{selected.name}` algorithm, seed, training rows, and validation rows.

- ROC-AUC with feature: `{selected.metrics['roc_auc']:.6f}`
- ROC-AUC without feature: `{ablation_metrics['roc_auc']:.6f}`
- Delta (with − without): `{selected.metrics['roc_auc'] - ablation_metrics['roc_auc']:.6f}`
- Feature importance/coefficient: `{cost_per_day_ablation['feature_importance_or_coefficient']}`
- Material improvement (>= 0.01 ROC-AUC and overall gate passed): `{cost_per_day_ablation['material_improvement']}`

The delta, not feature importance alone, answers whether it improves validated
performance. Importance describes model usage, not causality. Since the feature
is derived from `Shipping_Cost` and `Delivery_Days`, it may be redundant.

**Conclusion:** {cost_per_day_ablation['conclusion']}
"""
    (config.reports / "cost_per_day_ablation.md").write_text(ablation_md, encoding="utf-8")

    model_path = config.model_registry / "candidate_model.joblib"
    joblib.dump(selected.model, model_path)
    selection = {
        "generated_at": utc_now(),
        "selected_model": selected.name,
        "selection_reason": "Highest validation ROC-AUC among non-baseline candidates",
        "validation_metrics_at_default_threshold": selected.metrics,
        "validation_selected_threshold": threshold,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "mlflow_run_id": selected.mlflow_run_id,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "registry_stage": "candidate",
        "promotion_status": "pending final evaluation and business checkpoint",
    }
    write_json(config.reports / "analytical_model_selection.json", selection)
    write_json(config.model_registry / "candidate_manifest.json", selection)

    validation_signal_sufficient = selected.metrics["roc_auc"] >= config.minimum_roc_auc
    model_assessment = {
        "generated_at": utc_now(),
        "selected_model": selected.name,
        "validation_metrics": selected.metrics,
        "model_performance_sufficient": validation_signal_sufficient,
        "further_experiments_required": not validation_signal_sufficient,
        "cost_per_day_ablation": cost_per_day_ablation,
        "recommended_data_modification": (
            None
            if validation_signal_sufficient
            else "Collect predictive operational events available before the outcome, then revisit the target."
        ),
    }
    write_json(config.reports / "analytical_model_assessment_report.json", model_assessment)

    architecture = {
        "generated_at": utc_now(),
        "stages": [
            "raw CSV acquisition and schema validation",
            "data-quality gate",
            "feature engineering and train-only scaling",
            "feature-store publication",
            "candidate training and metadata logging",
            "validation-only selection and threshold tuning",
            "untouched-test evaluation",
            "business-goal checkpoint",
        ],
        "entrypoint": "python scripts/run_bpmn_pipeline.py",
        "source_repository": "this Git/project repository",
    }
    write_json(config.reports / "analytical_pipeline_architecture.json", architecture)
    return {
        "selection": selection,
        "assessment": model_assessment,
        "validation_comparison": comparison.to_dict("records"),
    }
