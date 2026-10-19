import os

from .config import PipelineConfig
from .io_utils import utc_now, write_json


def document_business_understanding(config: PipelineConfig) -> dict:
    """Create the management artifacts required before data work begins."""
    config.reports.mkdir(parents=True, exist_ok=True)
    assessment = {
        "generated_at": utc_now(),
        "problem": "Delayed orders create customer-service effort, recovery cost, and poor customer experience.",
        "decision_to_support": "Identify shipments likely to be delayed early enough for an operational intervention.",
        "users": ["shipping operations", "customer service", "project management"],
        "current_process": "Shipment outcomes are reviewed after delivery; the prototype tests earlier risk prediction.",
        "suitability": (
            "Suitable as an MLOps learning use case. Business deployment is conditional on measurable "
            "predictive and economic value."
        ),
        "continuation_decision": "CONTINUE_TO_DATA_COLLECTION",
    }
    requirements = {
        "generated_at": utc_now(),
        "business_objective": "Predict whether an order will be Delivered or Delayed from attributes known when the order is placed.",
        "functional_requirements": [
            "accept one or multiple shipment records",
            "return prediction, label, probability, and threshold",
            "record model and pipeline metadata",
            "expose health and monitoring information",
        ],
        "nonfunctional_requirements": [
            "reproducible training from raw CSV",
            "no train/validation/test leakage",
            "versioned model and feature artifacts",
            "automated tests and CI",
            "gated promotion with rollback protection",
        ],
        "success_criteria": {
            "minimum_test_roc_auc": config.minimum_roc_auc,
            "minimum_minority_recall": config.minimum_minority_recall,
            "minimum_weighted_f1_uplift": config.minimum_f1_improvement,
            "positive_candidate_net_benefit_required": True,
        },
    }
    project_map = {
        "generated_at": utc_now(),
        "project": "E-Commerce Shipping Prediction",
        "scope": "End-to-end prototype following the MLOps-based Data Science Process Model",
        "stages": [
            "Business Understanding",
            "Data Collection, Exploration, and Preparation",
            "Analysis",
            "Evaluation",
            "Deployment",
            "Utilization",
        ],
        "roles": {
            "strategy_and_project_management": {
                "owner": os.getenv("DSPM_STRATEGY_OWNER", "unassigned"),
                "responsibility": "business goals, checkpoint and continuation decisions",
                "stages": ["Business Understanding", "Evaluation", "Utilization"],
            },
            "data_management_and_analysis": {
                "owner": os.getenv("DSPM_DATA_OWNER", "unassigned"),
                "responsibility": "data requirements, acquisition, quality and feature pipeline",
                "stages": ["Data Collection, Exploration, and Preparation"],
            },
            "analytical_modeling": {
                "owner": os.getenv("DSPM_MODELING_OWNER", "unassigned"),
                "responsibility": "hypotheses, experiments, model registry and analytical pipeline",
                "stages": ["Analysis", "Evaluation"],
            },
            "domain_expertise": {
                "owner": os.getenv("DSPM_DOMAIN_OWNER", "unassigned"),
                "responsibility": "target interpretation, outcome feedback and business validation",
                "stages": ["Business Understanding", "Evaluation", "Utilization"],
            },
            "data_infrastructure_and_operations": {
                "owner": os.getenv("DSPM_OPERATIONS_OWNER", "unassigned"),
                "responsibility": "CI/CD, serving, monitoring, support and retraining",
                "stages": ["Deployment", "Utilization"],
            },
        },
        "project_plan": [
            {"stage": stage, "deliverable": deliverable, "exit_checkpoint": checkpoint}
            for stage, deliverable, checkpoint in [
                ("Business Understanding", "requirements, situation assessment and project map", "CONTINUE_TO_DATA_COLLECTION"),
                ("Data Collection, Exploration, and Preparation", "validated dataset and versioned feature artifacts", "DATA_QUALITY_PASSED"),
                ("Analysis", "reproducible candidate experiments and registered bundle", "MODEL_SIGNAL_SUFFICIENT"),
                ("Evaluation", "untouched-test assessment and business decision", "PROCEED_TO_DEPLOYMENT or RETURN_TO_BUSINESS_UNDERSTANDING"),
                ("Deployment", "approved pipeline and serving release", "DEPLOYMENT_SMOKE_TEST_PASSED"),
                ("Utilization", "user feedback, support and observed value", "CONTINUE, RETRAIN, or DISCONTINUE"),
            ]
        ],
        "checkpoint_policy": "Automated thresholds recommend continuation; a named business owner must review before operational use.",
        "primary_risk": "The provided fields may not contain enough pre-outcome predictive signal.",
    }
    write_json(config.reports / "situation_assessment.json", assessment)
    write_json(config.reports / "business_requirements.json", requirements)
    write_json(config.reports / "data_science_project_map.json", project_map)
    write_json(
        config.reports / "business_checkpoint.json",
        {
            "generated_at": utc_now(),
            "decision": assessment["continuation_decision"],
            "decision_role": "strategy_and_project_management",
            "owner": project_map["roles"]["strategy_and_project_management"]["owner"],
            "rationale": "The use case is suitable for a gated technical prototype; production remains conditional on measured predictive and economic value.",
            "human_review_required": True,
        },
    )
    return {"situation_assessment": assessment, "requirements": requirements, "project_map": project_map}
