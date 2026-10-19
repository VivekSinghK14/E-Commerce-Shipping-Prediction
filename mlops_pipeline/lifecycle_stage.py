from pathlib import Path
import json
import shutil

from .config import PipelineConfig, ROOT
from .io_utils import file_sha256, utc_now, write_json


OVERRIDE_STATUS = "approved_by_override"


def _exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def _active_override(config: PipelineConfig) -> dict | None:
    champion_path = config.model_registry / "champion_manifest.json"
    if not champion_path.exists():
        return None
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    if champion.get("promotion_status") != OVERRIDE_STATUS:
        return None
    if not (config.model_registry / champion.get("bundle_file", "champion_bundle.joblib")).exists():
        return None
    return champion["override"]


def _append_audit(config: PipelineConfig, entry: dict) -> None:
    config.artifacts.mkdir(parents=True, exist_ok=True)
    with (config.artifacts / "lifecycle_decision_log.jsonl").open("a", encoding="utf-8") as audit:
        audit.write(json.dumps(entry, sort_keys=True) + "\n")


def approve_deployment_override(config: PipelineConfig, actor: str, rationale: str) -> dict:
    """Deploy the evaluated candidate although its business gates failed.

    The evaluation result is never changed. The override is a separate,
    accountable decision: it names the actor, the rationale, and the failed
    gates, and the serving API reports it with every response.
    """
    if not actor.strip() or not rationale.strip():
        raise ValueError("actor and rationale are required for a deployment override")

    evaluation = json.loads((config.reports / "evaluation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (config.model_registry / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    candidate_bundle = config.model_registry / manifest.get("bundle_file", "candidate_bundle.joblib")
    if not candidate_bundle.exists():
        raise FileNotFoundError(f"Evaluated candidate bundle is missing: {candidate_bundle}")
    if file_sha256(candidate_bundle) != manifest["bundle_sha256"]:
        raise ValueError("Candidate bundle changed after evaluation; rerun the pipeline first")

    failed_gates = [
        name for name, check in evaluation["business_goal_checks"].items() if not check["passed"]
    ]
    test_metrics = evaluation["candidate_test_metrics"]
    override = {
        "approved_at": utc_now(),
        "actor": actor.strip(),
        "rationale": rationale.strip(),
        "model": manifest["selected_model"],
        "evaluation_decision": evaluation["checkpoint_decision"],
        "failed_business_gates": failed_gates,
        "test_metrics": {
            key: test_metrics[key]
            for key in ["roc_auc", "accuracy", "balanced_accuracy", "f1_weighted", "minority_recall"]
        },
        "candidate_net_benefit_eur": evaluation["candidate_business_economics"]["net_benefit_eur"],
        "scope": "Prototype demonstration only; not evidence that the model creates business value.",
    }

    champion_bundle = config.model_registry / "champion_bundle.joblib"
    shutil.copy2(candidate_bundle, champion_bundle)
    champion = {
        **manifest,
        "bundle_file": champion_bundle.name,
        "bundle_sha256": file_sha256(champion_bundle),
        "threshold": manifest["validation_selected_threshold"],
        "registry_stage": "deployed_by_override",
        "promotion_status": OVERRIDE_STATUS,
        "promoted_at": override["approved_at"],
        "checkpoint_decision": evaluation["checkpoint_decision"],
        "override": override,
    }
    write_json(config.model_registry / "champion_manifest.json", champion)
    write_json(config.reports / "deployment_override.json", override)
    _append_audit(config, {"action": "DEPLOYMENT_OVERRIDE", **override})

    lifecycle_path = config.reports / "lifecycle_status.json"
    if lifecycle_path.exists():
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        assess_deployment_and_utilization(config, {"checkpoint_decision": lifecycle["evaluation_decision"]})
    return champion


def assess_deployment_and_utilization(config: PipelineConfig, evaluation: dict) -> dict:
    """Prepare deployment/utilization evidence without bypassing the evaluation gate."""
    approved = evaluation["checkpoint_decision"] == "PROCEED_TO_DEPLOYMENT"
    override = None if approved else _active_override(config)
    deployment_components = {
        "serving_component": _exists("deployment/app.py"),
        "container_definition": _exists("Dockerfile"),
        "ci_cd_workflow": _exists(".github/workflows/deployment-ci.yml"),
        "deployment_tests": _exists("tests/test_deployment_api.py"),
        "smoke_test": _exists("scripts/smoke_test_deployment.py"),
        "monitoring_endpoints": _exists("deployment/app.py"),
        "guarded_retraining": _exists("scripts/retrain_model.py"),
    }
    deployment = {
        "generated_at": utc_now(),
        "impact_assessment": {
            "managerial": "A prediction API could support operations, but weak discrimination could waste intervention effort.",
            "technical": "FastAPI, Docker, CI, monitoring, MLflow, registry, and retraining components are available.",
            "risk": "The evaluated candidate does not meet predictive or absolute economic gates.",
        },
        "components": deployment_components,
        "prototype_deployment_status": (
            "PREDICTION_SERVING_AVAILABLE"
            if approved
            else "PREDICTION_SERVING_AVAILABLE_BY_OVERRIDE"
            if override
            else "COMPONENTS_AVAILABLE_PREDICTION_SERVING_BLOCKED"
        ),
        "production_business_integration": (
            "APPROVED"
            if approved
            else "DEPLOYED_BY_MANAGEMENT_OVERRIDE"
            if override
            else "BLOCKED_BY_EVALUATION_GATE"
        ),
        "deployment_executed": False,
        "reason": (
            "All gates passed"
            if approved
            else f"Evaluation gates failed; deployed by override of {override['actor']}: {override['rationale']}"
            if override
            else "The process model requires return to Business Understanding when business goals fail."
        ),
        "override": override,
        "rollback_policy": "A candidate never replaces the champion until all gates and tests pass.",
    }
    utilization = {
        "generated_at": utc_now(),
        "domain_user_interface": "FastAPI Swagger UI at /docs for prototype demonstration",
        "monitoring": ["/health", "/monitoring", "/metrics", "/pipeline/status", "prediction JSONL log"],
        "support": "Validation errors and runtime prediction events are counted and logged.",
        "maintenance": "Versioned retraining checks data hash, model age, observed performance, and manual triggers.",
        "continuous_training": "Implemented but promotion remains gated by data, model, business, and test checks.",
        "production_utilization_status": (
            "ACTIVE"
            if approved
            else "ACTIVE_UNDER_OVERRIDE_MODEL_FAILED_EVALUATION"
            if override
            else "NOT_STARTED_MODEL_NOT_APPROVED"
        ),
    }
    lifecycle = {
        "generated_at": utc_now(),
        "evaluation_decision": evaluation["checkpoint_decision"],
        "deployment": deployment,
        "utilization": utilization,
        "final_status": (
            "END_TO_END_APPROVED"
            if approved
            else "DEPLOYED_BY_OVERRIDE_EVALUATION_GATES_FAILED"
            if override
            else "TECHNICAL_PROTOTYPE_COMPLETE_BUSINESS_DEPLOYMENT_BLOCKED"
        ),
    }
    write_json(config.reports / "deployment_readiness_report.json", deployment)
    write_json(config.reports / "utilization_and_maintenance_report.json", utilization)
    write_json(config.reports / "lifecycle_status.json", lifecycle)
    return lifecycle


def manage_system_lifecycle(
    config: PipelineConfig,
    action: str,
    actor: str,
    rationale: str,
) -> dict:
    """Record an accountable discontinuation or guarded resumption decision."""
    if not actor.strip() or not rationale.strip():
        raise ValueError("actor and rationale are required")
    if action not in {"discontinue", "resume", "status"}:
        raise ValueError("action must be discontinue, resume, or status")

    state_path = config.artifacts / "system_state.json"
    audit_path = config.artifacts / "lifecycle_decision_log.jsonl"
    current_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"status": "ACTIVE"}
    )
    if action == "status":
        return current_state

    if action == "resume":
        checkpoint_path = config.reports / "checkpoint_decision.json"
        champion_manifest_path = config.model_registry / "champion_manifest.json"
        checkpoint = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.exists()
            else {}
        )
        champion = (
            json.loads(champion_manifest_path.read_text(encoding="utf-8"))
            if champion_manifest_path.exists()
            else {}
        )
        champion_bundle = config.model_registry / champion.get(
            "bundle_file", "champion_bundle.joblib"
        )
        gate_passed = (
            checkpoint.get("checkpoint_decision") == "PROCEED_TO_DEPLOYMENT"
            and champion.get("promotion_status") == "approved"
        )
        overridden = champion.get("promotion_status") == OVERRIDE_STATUS
        if not (gate_passed or overridden) or not champion_bundle.exists():
            raise ValueError(
                "resume requires an approved champion bundle (passing checkpoint or recorded override)"
            )

    state = {
        "status": "DISCONTINUED" if action == "discontinue" else "ACTIVE",
        "updated_at": utc_now(),
        "actor": actor.strip(),
        "rationale": rationale.strip(),
        "action": action.upper(),
    }
    config.artifacts.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    with audit_path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(state, sort_keys=True) + "\n")
    return state
