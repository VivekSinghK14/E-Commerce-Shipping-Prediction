import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import PipelineConfig
from .data_stage import collect_and_prepare
from .evaluation_stage import evaluate_and_checkpoint
from .io_utils import file_sha256, utc_now, write_json
from .model_stage import develop_and_assess_models


def retraining_reasons(
    config: PipelineConfig,
    force: bool = False,
    observed_roc_auc: float | None = None,
    max_age_days: int = 30,
) -> list[str]:
    state_path = config.artifacts / "retraining_state.json"
    if not state_path.exists():
        reasons = ["no_previous_retraining_state"]
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        reasons = []
        if state.get("source_sha256") != file_sha256(config.raw_data):
            reasons.append("source_data_changed")
        completed = datetime.fromisoformat(state["completed_at"])
        age_days = (datetime.now(timezone.utc) - completed).total_seconds() / 86400
        if age_days >= max_age_days:
            reasons.append("scheduled_model_age_reached")
    if observed_roc_auc is not None and observed_roc_auc < config.minimum_roc_auc:
        reasons.append("observed_performance_below_gate")
    if force:
        reasons.append("manual_force")
    return reasons


def run_retraining(
    config: PipelineConfig,
    force: bool = False,
    quick: bool = False,
    observed_roc_auc: float | None = None,
    max_age_days: int = 30,
) -> dict:
    """Run a versioned retraining attempt and promote only after all gates pass."""
    reasons = retraining_reasons(config, force, observed_roc_auc, max_age_days)
    if not reasons:
        report = {
            "generated_at": utc_now(),
            "status": "skipped",
            "reason": "No data-change, schedule, performance, or manual trigger",
        }
        write_json(config.artifacts / "retraining_report.json", report)
        return report

    run_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_artifacts = config.retraining_runs / run_version
    run_config = PipelineConfig(
        raw_data=config.raw_data,
        artifacts=run_artifacts,
        random_state=config.random_state,
        test_size=config.test_size,
        validation_size=config.validation_size,
        minimum_roc_auc=config.minimum_roc_auc,
        minimum_minority_recall=config.minimum_minority_recall,
        minimum_f1_improvement=config.minimum_f1_improvement,
        minimum_net_benefit_improvement=config.minimum_net_benefit_improvement,
        minimum_candidate_net_benefit=config.minimum_candidate_net_benefit,
        benefit_correct_risk_flag=config.benefit_correct_risk_flag,
        benefit_correct_on_time=config.benefit_correct_on_time,
        cost_missed_risk=config.cost_missed_risk,
        cost_unnecessary_intervention=config.cost_unnecessary_intervention,
        mlflow_tracking_dir=config.mlflow_store,
    )
    collect_and_prepare(run_config)
    develop_and_assess_models(run_config, quick=quick)
    evaluation = evaluate_and_checkpoint(run_config)

    promoted = evaluation["checkpoint_decision"] == "PROCEED_TO_DEPLOYMENT"
    champion_path = config.model_registry / "champion_model.joblib"
    if promoted:
        config.model_registry.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_config.model_registry / "candidate_model.joblib", champion_path)
        champion_bundle = config.model_registry / "champion_bundle.joblib"
        shutil.copy2(run_config.model_registry / "candidate_bundle.joblib", champion_bundle)
        candidate_manifest = json.loads(
            (run_config.model_registry / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        candidate_manifest.update(
            {
                "bundle_file": "champion_bundle.joblib",
                "promotion_status": "approved",
                "promoted_at": utc_now(),
                "checkpoint_decision": evaluation["checkpoint_decision"],
            }
        )
        write_json(config.model_registry / "champion_manifest.json", candidate_manifest)

    config.reports.mkdir(parents=True, exist_ok=True)
    write_json(config.reports / "checkpoint_decision.json", evaluation)

    report = {
        "generated_at": utc_now(),
        "status": "completed",
        "version": run_version,
        "triggers": reasons,
        "run_artifacts": str(run_artifacts),
        "checkpoint_decision": evaluation["checkpoint_decision"],
        "promoted": promoted,
        "champion_path": str(champion_path) if promoted else None,
        "champion_bundle_path": (
            str(config.model_registry / "champion_bundle.joblib") if promoted else None
        ),
        "rollback": (
            "Previous champion is unchanged until the new candidate passes every gate; "
            "versioned run artifacts remain available for audit."
        ),
    }
    write_json(config.artifacts / "retraining_report.json", report)
    write_json(
        config.artifacts / "retraining_state.json",
        {
            "completed_at": utc_now(),
            "source_sha256": file_sha256(config.raw_data),
            "last_version": run_version,
            "last_decision": evaluation["checkpoint_decision"],
        },
    )
    return report
