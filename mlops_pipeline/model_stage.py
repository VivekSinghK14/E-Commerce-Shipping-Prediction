import json
import time
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .business import classification_economics, model_business_characteristics
from .config import FEATURE_COLUMNS, NUMERIC_COLUMNS, PipelineConfig
from .io_utils import file_sha256, utc_now, write_json
from .mlflow_utils import configure_mlflow


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
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    delayed_true = 1 - np.asarray(y_true)
    delayed_probability = 1 - np.asarray(probability)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_weighted": float(f1_score(y_true, prediction, average="weighted")),
        "precision_weighted": float(
            precision_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, prediction, average="weighted")),
        "minority_recall": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "minority_precision": float(
            precision_score(y_true, prediction, pos_label=0, zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision_delayed": float(
            average_precision_score(delayed_true, delayed_probability)
        ),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
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


def _tune_threshold(
    y_true: pd.Series, probabilities: np.ndarray, config: PipelineConfig
) -> tuple[float, list[dict]]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        prediction = (probabilities >= threshold).astype(int)
        economics = classification_economics(y_true, prediction, config)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
                "f1_weighted": float(f1_score(y_true, prediction, average="weighted")),
                "minority_recall": float(
                    recall_score(y_true, prediction, pos_label=0, zero_division=0)
                ),
                "net_benefit_eur": economics["net_benefit_eur"],
                "candidate_net_benefit_positive": economics["net_benefit_eur"] > 0,
            }
        )
    # Threshold choice follows the stated business objective. Minority recall is
    # treated as a safety constraint so a profitable majority-only policy cannot win.
    recall_feasible = [
        row for row in rows if row["minority_recall"] >= config.minimum_minority_recall
    ]
    business_feasible = [
        row for row in recall_feasible if row["candidate_net_benefit_positive"]
    ]
    if business_feasible:
        winner = max(
            business_feasible,
            key=lambda row: (
                row["net_benefit_eur"],
                row["balanced_accuracy"],
                row["f1_weighted"],
            ),
        )
    else:
        # No threshold meets the economic gate. Keep a diagnostically balanced
        # threshold instead of pretending the least-negative policy is viable.
        pool = recall_feasible or rows
        winner = max(
            pool,
            key=lambda row: (
                row["balanced_accuracy"],
                row["f1_weighted"],
                row["net_benefit_eur"],
            ),
        )
    return winner["threshold"], rows


def develop_and_assess_models(config: PipelineConfig, quick: bool = False) -> dict:
    """Implement test design, model selection/development, and analytical assessment."""
    x_train, y_train = _load_split(config, "train")
    x_validation, y_validation = _load_split(config, "validation")
    config.metadata_store.mkdir(parents=True, exist_ok=True)
    config.model_registry.mkdir(parents=True, exist_ok=True)
    configure_mlflow(config)

    hypotheses = {
        "generated_at": utc_now(),
        "business_hypothesis": (
            "Attributes available at or near order time contain enough signal to identify "
            "shipments that will be delayed (Delivered vs Delayed)."
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
        "problem": "Binary classification: Delivered (1) vs Delayed (0)",
        "selection_dataset": "validation split only",
        "final_assessment_dataset": "untouched test split only",
        "selection_metric": "validation ROC-AUC",
        "threshold_metric": (
            "maximum positive validation net benefit subject to the minority-recall gate; "
            "if no threshold is economically viable, use balanced accuracy as a diagnostic fallback"
        ),
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
            input_example = x_train.head(5).astype(float)
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
    baseline_result = next(result for result in results if result.name == "dummy_majority")
    candidate_auc_uplift = selected.metrics["roc_auc"] - baseline_result.metrics["roc_auc"]
    validation_probability = selected.model.predict_proba(x_validation)[:, 1]
    threshold, threshold_rows = _tune_threshold(
        y_validation, validation_probability, config
    )
    selected_threshold_row = next(
        row for row in threshold_rows if row["threshold"] == threshold
    )
    pd.DataFrame(threshold_rows).to_csv(
        config.reports / "threshold_sweep_validation.csv", index=False
    )

    model_path = config.model_registry / "candidate_model.joblib"
    joblib.dump(selected.model, model_path)
    preparation = json.loads(
        (config.feature_store / "feature_manifest.json").read_text(encoding="utf-8")
    )
    bundle_path = config.model_registry / "candidate_bundle.joblib"
    joblib.dump(
        {
            "model": selected.model,
            "scaler": joblib.load(config.feature_store / "scaler.joblib"),
            "category_maps": preparation["category_maps"],
            "feature_columns": FEATURE_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
            "threshold": threshold,
            "model_name": selected.name,
        },
        bundle_path,
    )
    selection = {
        "generated_at": utc_now(),
        "selected_model": selected.name,
        "selection_role": "best experimental non-baseline candidate",
        "selection_reason": "Highest validation ROC-AUC among non-baseline candidates",
        "best_overall_validation_model": comparison.iloc[0]["model"],
        "naive_baseline_model": baseline_result.name,
        "naive_baseline_validation_metrics": baseline_result.metrics,
        "validation_roc_auc_uplift_over_naive_baseline": candidate_auc_uplift,
        "candidate_beats_naive_baseline_auc": candidate_auc_uplift > 0,
        "validation_metrics_at_default_threshold": selected.metrics,
        "validation_selected_threshold": threshold,
        "threshold_selection": {
            "dataset": "validation",
            "policy": test_design["threshold_metric"],
            "status": (
                "business_threshold_selected"
                if selected_threshold_row["candidate_net_benefit_positive"]
                else "no_economically_viable_threshold_diagnostic_fallback"
            ),
            "metrics": selected_threshold_row,
        },
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "bundle_file": "candidate_bundle.joblib",
        "bundle_path": str(bundle_path),
        "bundle_sha256": file_sha256(bundle_path),
        "feature_columns": FEATURE_COLUMNS,
        "mlflow_run_id": selected.mlflow_run_id,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "registry_stage": "candidate",
        "promotion_status": "pending final evaluation and business checkpoint",
    }
    write_json(config.reports / "analytical_model_selection.json", selection)
    write_json(config.model_registry / "candidate_manifest.json", selection)

    validation_signal_sufficient = (
        selected.metrics["roc_auc"] >= config.minimum_roc_auc
        and candidate_auc_uplift > 0
    )
    model_assessment = {
        "generated_at": utc_now(),
        "selected_model": selected.name,
        "validation_metrics": selected.metrics,
        "model_performance_sufficient": validation_signal_sufficient,
        "candidate_beats_naive_baseline_auc": candidate_auc_uplift > 0,
        "validation_roc_auc_uplift_over_naive_baseline": candidate_auc_uplift,
        "further_experiments_required": not validation_signal_sufficient,
        "leakage_assessment": json.loads(
            (config.reports / "feature_leakage_assessment.json").read_text(encoding="utf-8")
        ),
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
