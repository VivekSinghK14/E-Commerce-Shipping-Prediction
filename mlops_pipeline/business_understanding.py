from .config import PipelineConfig
from .io_utils import utc_now, write_json


def document_business_understanding(config: PipelineConfig) -> dict:
    """Create the management artifacts required before data work begins."""
    config.reports.mkdir(parents=True, exist_ok=True)
    assessment = {
        "generated_at": utc_now(),
        "problem": "Late or returned orders create customer-service effort, recovery cost, and poor customer experience.",
        "decision_to_support": "Identify at-risk shipments early enough for an operational intervention.",
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
        "business_objective": "Predict whether an order will be delivered/on time from available inputs.",
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
            "strategy_and_project_management": "business goals, checkpoint and continuation decisions",
            "data_management_and_analysis": "requirements, data acquisition, quality and pipeline",
            "analytical_modeling": "hypotheses, experiments, models and analytical pipeline",
            "domain_expertise": "interpretation and business validation",
            "data_infrastructure_and_operations": "CI/CD, serving, monitoring and retraining",
        },
        "primary_risk": "The provided fields may not contain enough pre-outcome predictive signal.",
    }
    write_json(config.reports / "situation_assessment.json", assessment)
    write_json(config.reports / "business_requirements.json", requirements)
    write_json(config.reports / "data_science_project_map.json", project_map)
    return {"situation_assessment": assessment, "requirements": requirements, "project_map": project_map}
