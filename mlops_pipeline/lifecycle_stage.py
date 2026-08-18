from pathlib import Path

from .config import PipelineConfig, ROOT
from .io_utils import utc_now, write_json


def _exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def assess_deployment_and_utilization(config: PipelineConfig, evaluation: dict) -> dict:
    """Prepare deployment/utilization evidence without bypassing the evaluation gate."""
    approved = evaluation["checkpoint_decision"] == "PROCEED_TO_DEPLOYMENT"
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
        "prototype_deployment_status": "AVAILABLE_FOR_TECHNICAL_DEMONSTRATION",
        "production_business_integration": "APPROVED" if approved else "BLOCKED_BY_EVALUATION_GATE",
        "deployment_executed": False,
        "reason": (
            "All gates passed"
            if approved
            else "The process model requires return to Business Understanding when business goals fail."
        ),
        "rollback_policy": "A candidate never replaces the champion until all gates and tests pass.",
    }
    utilization = {
        "generated_at": utc_now(),
        "domain_user_interface": "FastAPI Swagger UI at /docs for prototype demonstration",
        "monitoring": ["/health", "/monitoring", "/metrics", "/pipeline/status", "prediction JSONL log"],
        "support": "Validation errors and runtime prediction events are counted and logged.",
        "maintenance": "Versioned retraining checks data hash, model age, observed performance, and manual triggers.",
        "continuous_training": "Implemented but promotion remains gated by data, model, business, and test checks.",
        "production_utilization_status": "ACTIVE" if approved else "NOT_STARTED_MODEL_NOT_APPROVED",
    }
    lifecycle = {
        "generated_at": utc_now(),
        "evaluation_decision": evaluation["checkpoint_decision"],
        "deployment": deployment,
        "utilization": utilization,
        "final_status": (
            "END_TO_END_APPROVED"
            if approved
            else "TECHNICAL_PROTOTYPE_COMPLETE_BUSINESS_DEPLOYMENT_BLOCKED"
        ),
    }
    write_json(config.reports / "deployment_readiness_report.json", deployment)
    write_json(config.reports / "utilization_and_maintenance_report.json", utilization)
    write_json(config.reports / "lifecycle_status.json", lifecycle)
    return lifecycle
