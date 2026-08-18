import json

import joblib
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier

from .business import classification_economics
from .config import PipelineConfig
from .io_utils import file_sha256, utc_now, write_json
from .model_stage import _metrics


def evaluate_and_checkpoint(config: PipelineConfig) -> dict:
    """Assess analytical results, create reports, and make the BPMN checkpoint decision."""
    selection = json.loads(
        (config.reports / "analytical_model_selection.json").read_text(encoding="utf-8")
    )
    x_train = pd.read_csv(config.feature_store / "X_train.csv")
    y_train = pd.read_csv(config.feature_store / "y_train.csv")["binary_target"]
    x_test = pd.read_csv(config.feature_store / "X_test.csv")
    y_test = pd.read_csv(config.feature_store / "y_test.csv")["binary_target"]

    model = joblib.load(config.model_registry / "candidate_model.joblib")
    threshold = float(selection["validation_selected_threshold"])
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    candidate_metrics = _metrics(y_test, prediction, probability)

    baseline = DummyClassifier(strategy="prior").fit(x_train, y_train)
    baseline_probability = baseline.predict_proba(x_test)[:, 1]
    baseline_prediction = baseline.predict(x_test)
    baseline_metrics = _metrics(y_test, baseline_prediction, baseline_probability)
    candidate_economics = classification_economics(y_test, prediction, config)
    baseline_economics = classification_economics(y_test, baseline_prediction, config)
    net_benefit_improvement = (
        candidate_economics["net_benefit_eur"] - baseline_economics["net_benefit_eur"]
    )

    checks = {
        "roc_auc": {
            "actual": candidate_metrics["roc_auc"],
            "required": config.minimum_roc_auc,
            "passed": candidate_metrics["roc_auc"] >= config.minimum_roc_auc,
        },
        "minority_recall": {
            "actual": candidate_metrics["minority_recall"],
            "required": config.minimum_minority_recall,
            "passed": candidate_metrics["minority_recall"] >= config.minimum_minority_recall,
        },
        "weighted_f1_improvement_over_baseline": {
            "actual": candidate_metrics["f1_weighted"] - baseline_metrics["f1_weighted"],
            "required": config.minimum_f1_improvement,
            "passed": (
                candidate_metrics["f1_weighted"] - baseline_metrics["f1_weighted"]
                >= config.minimum_f1_improvement
            ),
        },
        "net_benefit_improvement_over_baseline_eur": {
            "actual": net_benefit_improvement,
            "required": config.minimum_net_benefit_improvement,
            "passed": net_benefit_improvement > config.minimum_net_benefit_improvement,
        },
        "candidate_net_benefit_eur": {
            "actual": candidate_economics["net_benefit_eur"],
            "required": config.minimum_candidate_net_benefit,
            "passed": candidate_economics["net_benefit_eur"] > config.minimum_candidate_net_benefit,
        },
    }
    goals_satisfied = all(check["passed"] for check in checks.values())
    decision = "PROCEED_TO_DEPLOYMENT" if goals_satisfied else "RETURN_TO_BUSINESS_UNDERSTANDING"

    evaluation = {
        "generated_at": utc_now(),
        "model": selection["selected_model"],
        "model_sha256": file_sha256(config.model_registry / "candidate_model.joblib"),
        "test_set_policy": "Used once after model and threshold were frozen on validation",
        "threshold": threshold,
        "candidate_test_metrics": candidate_metrics,
        "naive_baseline_test_metrics": baseline_metrics,
        "candidate_business_economics": candidate_economics,
        "naive_baseline_business_economics": baseline_economics,
        "net_benefit_improvement_over_baseline_eur": net_benefit_improvement,
        "business_goal_checks": checks,
        "business_goals_satisfied": goals_satisfied,
        "checkpoint_decision": decision,
        "next_action": (
            "Register/promote for deployment review"
            if goals_satisfied
            else "Revisit target, available predictors, and business framing; do not deploy"
        ),
    }
    config.mlflow_store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_store.resolve().as_uri())
    mlflow.set_experiment("ecommerce-shipping-bpmn")
    with mlflow.start_run(run_name=f"final_evaluation_{selection['selected_model']}") as run:
        mlflow.set_tags(
            {
                "pipeline_stage": "evaluation",
                "checkpoint_decision": decision,
                "candidate_training_run_id": selection.get("mlflow_run_id", "unknown"),
            }
        )
        mlflow.log_param("selected_model", selection["selected_model"])
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics(
            {
                **{f"test_{key}": value for key, value in candidate_metrics.items()},
                "baseline_f1_weighted": baseline_metrics["f1_weighted"],
                "candidate_net_benefit_eur": candidate_economics["net_benefit_eur"],
                "baseline_net_benefit_eur": baseline_economics["net_benefit_eur"],
                "net_benefit_improvement_eur": net_benefit_improvement,
            }
        )
        evaluation["mlflow_evaluation_run_id"] = run.info.run_id
    write_json(config.reports / "final_model_report.json", evaluation)
    write_json(config.reports / "evaluation_report.json", evaluation)
    write_json(config.reports / "checkpoint_decision.json", evaluation)

    insights = f"""# Modeling insights

- Selected candidate: `{selection['selected_model']}` (selected using validation ROC-AUC).
- Frozen validation threshold: `{threshold:.2f}`.
- Untouched-test ROC-AUC: `{candidate_metrics['roc_auc']:.4f}`; chance is `0.5000`.
- Candidate weighted F1: `{candidate_metrics['f1_weighted']:.4f}`.
- Naive majority weighted F1: `{baseline_metrics['f1_weighted']:.4f}`.
- Minority-class recall: `{candidate_metrics['minority_recall']:.4f}`.
- Estimated candidate net benefit: `EUR {candidate_economics['net_benefit_eur']:.2f}`.
- Estimated baseline net benefit: `EUR {baseline_economics['net_benefit_eur']:.2f}`.

The available features do not demonstrate reliable predictive signal. The result
supports the null hypothesis and does not justify model promotion.
"""
    (config.reports / "modeling_insights.md").write_text(insights, encoding="utf-8")

    failed_goals = [name for name, check in checks.items() if not check["passed"]]
    human_report = f"""# Evaluation report

## Assessment

The model and threshold were frozen using validation data. The test split was
then used once for final assessment.

- Candidate: `{selection['selected_model']}`
- Test ROC-AUC: `{candidate_metrics['roc_auc']:.4f}`
- Test weighted F1: `{candidate_metrics['f1_weighted']:.4f}`
- Baseline weighted F1: `{baseline_metrics['f1_weighted']:.4f}`
- Test minority recall: `{candidate_metrics['minority_recall']:.4f}`
- Candidate net benefit: `EUR {candidate_economics['net_benefit_eur']:.2f}`
- Baseline net benefit: `EUR {baseline_economics['net_benefit_eur']:.2f}`
- Failed business gates: `{', '.join(failed_goals) if failed_goals else 'none'}`

## Checkpoint decision

**{decision}**

{evaluation['next_action']}.
"""
    (config.reports / "final_model_report.md").write_text(human_report, encoding="utf-8")
    (config.reports / "evaluation_report.md").write_text(human_report, encoding="utf-8")

    cost_report = {
        "generated_at": utc_now(),
        "scenario": (
            "Correct risk flags enable intervention; missed delayed/returned shipments are the "
            "costliest error; false alarms incur an intervention cost."
        ),
        "candidate": candidate_economics,
        "baseline": baseline_economics,
        "candidate_minus_baseline_eur": net_benefit_improvement,
        "decision_use": "Scenario analysis; replace assumed unit values with validated business data.",
    }
    write_json(config.reports / "cost_benefit_analysis.json", cost_report)
    cost_md = f"""# Cost and benefit analysis

The positive class is **Delivered**. Therefore a true negative is a correctly
identified delayed/returned shipment and a false positive is a missed risk.

| Classification outcome | Candidate count | Business treatment | Unit value |
|---|---:|---|---:|
| Correct risk flag (TN) | {candidate_economics['confusion_matrix']['tn']} | intervention opportunity | EUR {config.benefit_correct_risk_flag:.2f} benefit |
| Correct on-time (TP) | {candidate_economics['confusion_matrix']['tp']} | avoid unnecessary action | EUR {config.benefit_correct_on_time:.2f} benefit |
| Missed risk (FP) | {candidate_economics['confusion_matrix']['fp']} | delay/return not flagged | EUR {config.cost_missed_risk:.2f} cost |
| Unnecessary intervention (FN) | {candidate_economics['confusion_matrix']['fn']} | on-time order incorrectly flagged | EUR {config.cost_unnecessary_intervention:.2f} cost |

- Candidate estimated net benefit: **EUR {candidate_economics['net_benefit_eur']:.2f}**
- Baseline estimated net benefit: **EUR {baseline_economics['net_benefit_eur']:.2f}**
- Candidate minus baseline: **EUR {net_benefit_improvement:.2f}**

These unit values are explicit assumptions for scenario comparison. Finance and
operations must replace them with verified values before a real investment decision.
"""
    (config.reports / "cost_benefit_analysis.md").write_text(cost_md, encoding="utf-8")

    pros = [
        "The pipeline is reproducible and the test set is isolated from selection.",
        "The chosen threshold identifies some at-risk shipments.",
        "All business assumptions and failed gates are explicit and auditable.",
    ]
    cons = [
        "ROC-AUC is approximately chance, so ranking is unreliable.",
        "The candidate does not improve weighted F1 over the majority baseline.",
        "Minority recall and economic value depend heavily on threshold and assumed costs.",
        "Current fields contain little pre-outcome signal; deployment could create cost without benefit.",
    ]
    objective_assessment = {
        "generated_at": utc_now(),
        "objective": "Identify delayed/returned shipments early enough for useful intervention.",
        "pros": pros,
        "cons": cons,
        "conclusion": decision,
        "recommendation": evaluation["next_action"],
    }
    write_json(config.reports / "business_objective_assessment.json", objective_assessment)
    objective_md = "# Business objective assessment\n\n## Pros\n\n" + "\n".join(
        f"- {item}" for item in pros
    ) + "\n\n## Cons\n\n" + "\n".join(f"- {item}" for item in cons) + (
        f"\n\n## Conclusion\n\n**{decision}** — {evaluation['next_action']}.\n"
    )
    (config.reports / "business_objective_assessment.md").write_text(
        objective_md, encoding="utf-8"
    )

    # Attach final business artifacts to the MLflow evaluation run.
    with mlflow.start_run(run_id=evaluation["mlflow_evaluation_run_id"]):
        for artifact_name in [
            "evaluation_report.json",
            "evaluation_report.md",
            "cost_benefit_analysis.json",
            "cost_benefit_analysis.md",
            "business_objective_assessment.json",
            "business_objective_assessment.md",
        ]:
            mlflow.log_artifact(str(config.reports / artifact_name), artifact_path="reports")

    manifest_path = config.model_registry / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_evaluation"] = candidate_metrics
    manifest["business_goals_satisfied"] = goals_satisfied
    manifest["checkpoint_decision"] = decision
    manifest["registry_stage"] = "approved" if goals_satisfied else "rejected"
    manifest["promotion_status"] = "eligible" if goals_satisfied else "blocked"
    write_json(manifest_path, manifest)
    return evaluation
