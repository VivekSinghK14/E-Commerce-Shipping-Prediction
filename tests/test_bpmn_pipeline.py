import json

import pandas as pd

from mlops_pipeline.config import FEATURE_COLUMNS, PipelineConfig
from mlops_pipeline.business import classification_economics
from mlops_pipeline.business_understanding import document_business_understanding
from mlops_pipeline.data_stage import collect_and_prepare
from mlops_pipeline.lifecycle_stage import assess_deployment_and_utilization
from mlops_pipeline.retraining import retraining_reasons


def test_data_stage_creates_all_bpmn_reports_and_leakage_safe_splits(tmp_path):
    config = PipelineConfig(artifacts=tmp_path / "artifacts")

    result = collect_and_prepare(config)

    assert result["quality"]["data_sufficient"] is True
    assert (config.reports / "data_description_report.json").exists()
    assert (config.reports / "data_dictionary.csv").exists()
    assert (config.reports / "data_exploration_report.json").exists()
    assert (config.reports / "data_quality_report.json").exists()
    assert (config.reports / "data_preparation_report.json").exists()
    assert (config.reports / "data_requirements.json").exists()
    assert (config.reports / "data_pipeline_architecture.json").exists()
    assert (config.reports / "input_data_contract.json").exists()
    assert (config.reports / "output_data_contract.json").exists()
    assert (config.feature_store / "feature_manifest.json").exists()

    raw = pd.read_csv(config.raw_data)
    in_scope_rows = int(raw["Delivery_Status"].isin(["Delivered", "Delayed"]).sum())

    split_sizes = {}
    for split in ["train", "validation", "test"]:
        features = pd.read_csv(config.feature_store / f"X_{split}.csv")
        targets = pd.read_csv(config.feature_store / f"y_{split}.csv")
        assert list(features.columns) == FEATURE_COLUMNS
        assert list(targets.columns) == ["binary_target"]
        assert set(targets["binary_target"]) == {0, 1}
        assert len(features) == len(targets)
        split_sizes[split] = len(features)

    assert sum(split_sizes.values()) == in_scope_rows
    assert abs(split_sizes["test"] / in_scope_rows - 0.20) < 0.001
    assert abs(split_sizes["validation"] / in_scope_rows - 0.20) < 0.001
    manifest = json.loads(
        (config.feature_store / "feature_manifest.json").read_text(encoding="utf-8")
    )
    assert "scaler fit on training rows only" in manifest["leakage_controls"]
    assert "Ship_Date" not in manifest["feature_columns"]
    assert "Delivery_Date" not in manifest["feature_columns"]
    assert "Delivery_Days" not in manifest["feature_columns"]
    assert "Cost_per_Day" not in manifest["feature_columns"]


def test_returned_orders_are_removed_before_modeling(tmp_path):
    config = PipelineConfig(artifacts=tmp_path / "artifacts")
    raw = pd.read_csv(config.raw_data)

    collect_and_prepare(config)

    preparation = json.loads(
        (config.reports / "data_preparation_report.json").read_text(encoding="utf-8")
    )
    scope = preparation["scope_filter"]
    assert scope["kept_outcomes"] == ["Delivered", "Delayed"]
    assert scope["removed_outcomes"] == {"Returned": int((raw["Delivery_Status"] == "Returned").sum())}
    assert scope["rows_in_scope"] + scope["rows_removed"] == len(raw)
    assert preparation["target_maps"] == {"binary": {"Delivered": 1, "Delayed": 0}}

    exploration = json.loads(
        (config.reports / "data_exploration_report.json").read_text(encoding="utf-8")
    )
    assert set(exploration["category_distributions"]["Delivery_Status"]) == {"Delivered", "Delayed"}
    contract = json.loads(
        (config.reports / "input_data_contract.json").read_text(encoding="utf-8")
    )
    assert contract["columns"]["Delivery_Status"]["allowed"] == ["Delivered", "Delayed"]


def test_data_stage_uses_only_delivered_and_delayed_from_a_custom_source(tmp_path):
    raw = pd.read_csv(PipelineConfig().raw_data).head(3000)
    source = tmp_path / "sample.csv"
    raw.to_csv(source, index=False)
    config = PipelineConfig(raw_data=source, artifacts=tmp_path / "artifacts")

    collect_and_prepare(config)

    expected = int(raw["Delivery_Status"].isin(["Delivered", "Delayed"]).sum())
    rows = sum(
        len(pd.read_csv(config.feature_store / f"y_{split}.csv"))
        for split in ["train", "validation", "test"]
    )
    assert rows == expected


def test_checked_in_evaluation_has_an_explicit_checkpoint():
    report_path = PipelineConfig().reports / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["checkpoint_decision"] in {
        "PROCEED_TO_DEPLOYMENT",
        "RETURN_TO_BUSINESS_UNDERSTANDING",
    }
    assert set(report["business_goal_checks"]) == {
        "roc_auc",
        "minority_recall",
        "weighted_f1_improvement_over_baseline",
        "net_benefit_improvement_over_baseline_eur",
        "candidate_net_benefit_eur",
    }
    assert report["mlflow_evaluation_run_id"]
    assert "candidate_business_economics" in report
    if report["checkpoint_decision"] == "RETURN_TO_BUSINESS_UNDERSTANDING":
        assert report["final_selected_model"] == report["best_experimental_candidate"]
        assert report["final_selection_status"] == "SELECTED_FOR_PROTOTYPE_NOT_APPROVED_FOR_DEPLOYMENT"
        assert report["deployment_model"] is None
    assert (PipelineConfig().reports / "naive_vs_best_candidate.png").exists()


def test_leakage_assessment_excludes_post_outcome_inputs(tmp_path):
    config = PipelineConfig(artifacts=tmp_path / "artifacts")
    collect_and_prepare(config)
    report_path = config.reports / "feature_leakage_assessment.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["prediction_time"] == "when the order is placed"
    assert "Delivery_Days" in report["excluded_post_outcome_fields"]
    assert "Cost_per_Day" in report["excluded_post_outcome_fields"]
    assert report["status"] == "passed"


def test_business_costs_distinguish_all_four_classification_outcomes():
    result = classification_economics(
        y_true=[0, 0, 1, 1], prediction=[0, 1, 0, 1], config=PipelineConfig()
    )

    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["net_benefit_eur"] == 30 + 2 - 80 - 10


def test_retraining_trigger_detects_missing_state(tmp_path):
    config = PipelineConfig(artifacts=tmp_path / "artifacts")

    assert retraining_reasons(config) == ["no_previous_retraining_state"]


def test_business_understanding_creates_required_management_artifacts(tmp_path):
    config = PipelineConfig(artifacts=tmp_path / "artifacts")

    result = document_business_understanding(config)

    assert result["situation_assessment"]["continuation_decision"] == "CONTINUE_TO_DATA_COLLECTION"
    assert (config.reports / "business_requirements.json").exists()
    assert (config.reports / "data_science_project_map.json").exists()


def test_failed_evaluation_blocks_business_deployment_but_keeps_prototype(tmp_path):
    config = PipelineConfig(artifacts=tmp_path / "artifacts")
    evaluation = {"checkpoint_decision": "RETURN_TO_BUSINESS_UNDERSTANDING"}

    lifecycle = assess_deployment_and_utilization(config, evaluation)

    assert lifecycle["deployment"]["production_business_integration"] == "BLOCKED_BY_EVALUATION_GATE"
    assert lifecycle["deployment"]["prototype_deployment_status"] == "COMPONENTS_AVAILABLE_PREDICTION_SERVING_BLOCKED"
    assert lifecycle["utilization"]["production_utilization_status"] == "NOT_STARTED_MODEL_NOT_APPROVED"


def _evaluated_registry(tmp_path):
    import joblib
    from mlops_pipeline.io_utils import file_sha256, write_json

    config = PipelineConfig(artifacts=tmp_path / "artifacts")
    config.model_registry.mkdir(parents=True)
    config.reports.mkdir(parents=True)
    bundle = config.model_registry / "candidate_bundle.joblib"
    joblib.dump({"model": "stub"}, bundle)
    write_json(
        config.model_registry / "candidate_manifest.json",
        {
            "selected_model": "xgboost",
            "bundle_file": bundle.name,
            "bundle_sha256": file_sha256(bundle),
            "validation_selected_threshold": 0.85,
        },
    )
    metrics = {"roc_auc": 0.49, "accuracy": 0.5, "balanced_accuracy": 0.49, "f1_weighted": 0.57, "minority_recall": 0.47}
    write_json(
        config.reports / "evaluation_report.json",
        {
            "checkpoint_decision": "RETURN_TO_BUSINESS_UNDERSTANDING",
            "business_goal_checks": {"roc_auc": {"passed": False}, "minority_recall": {"passed": True}},
            "candidate_test_metrics": metrics,
            "candidate_business_economics": {"net_benefit_eur": -1.0},
        },
    )
    assess_deployment_and_utilization(config, {"checkpoint_decision": "RETURN_TO_BUSINESS_UNDERSTANDING"})
    return config


def test_deployment_override_is_recorded_without_changing_evaluation(tmp_path):
    from mlops_pipeline.lifecycle_stage import approve_deployment_override

    config = _evaluated_registry(tmp_path)
    champion = approve_deployment_override(config, "Course owner", "Prototype must be demonstrated")

    assert champion["promotion_status"] == "approved_by_override"
    assert champion["override"]["failed_business_gates"] == ["roc_auc"]
    assert (config.model_registry / "champion_bundle.joblib").exists()
    evaluation = json.loads((config.reports / "evaluation_report.json").read_text(encoding="utf-8"))
    assert evaluation["checkpoint_decision"] == "RETURN_TO_BUSINESS_UNDERSTANDING"
    lifecycle = json.loads((config.reports / "lifecycle_status.json").read_text(encoding="utf-8"))
    assert lifecycle["final_status"] == "DEPLOYED_BY_OVERRIDE_EVALUATION_GATES_FAILED"
    assert lifecycle["deployment"]["production_business_integration"] == "DEPLOYED_BY_MANAGEMENT_OVERRIDE"
    assert "DEPLOYMENT_OVERRIDE" in (config.artifacts / "lifecycle_decision_log.jsonl").read_text(encoding="utf-8")


def test_deployment_override_requires_accountability_and_the_evaluated_bundle(tmp_path):
    import pytest
    from mlops_pipeline.lifecycle_stage import approve_deployment_override

    config = _evaluated_registry(tmp_path)
    with pytest.raises(ValueError):
        approve_deployment_override(config, " ", "reason")
    (config.model_registry / "candidate_bundle.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after evaluation"):
        approve_deployment_override(config, "Course owner", "reason")
