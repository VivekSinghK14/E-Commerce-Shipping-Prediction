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

    split_sizes = {}
    for split in ["train", "validation", "test"]:
        features = pd.read_csv(config.feature_store / f"X_{split}.csv")
        targets = pd.read_csv(config.feature_store / f"y_{split}.csv")
        assert list(features.columns) == FEATURE_COLUMNS
        assert len(features) == len(targets)
        split_sizes[split] = len(features)

    assert split_sizes == {"train": 30000, "validation": 10000, "test": 10000}
    manifest = json.loads(
        (config.feature_store / "feature_manifest.json").read_text(encoding="utf-8")
    )
    assert "scaler fit on training rows only" in manifest["leakage_controls"]
    assert "Ship_Date" not in manifest["feature_columns"]
    assert "Delivery_Date" not in manifest["feature_columns"]


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


def test_cost_per_day_has_a_controlled_ablation_conclusion():
    report_path = PipelineConfig().reports / "cost_per_day_ablation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "with_cost_per_day" in report
    assert "without_cost_per_day" in report
    assert report["material_improvement_threshold_roc_auc"] == 0.01
    assert report["conclusion"]


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
    assert lifecycle["deployment"]["prototype_deployment_status"] == "AVAILABLE_FOR_TECHNICAL_DEMONSTRATION"
    assert lifecycle["utilization"]["production_utilization_status"] == "NOT_STARTED_MODEL_NOT_APPROVED"
