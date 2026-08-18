import json
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
OUTPUT = ROOT / "output" / "pdf" / "ecommerce_mlops_complete_project_report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E75B6")
LIGHT_BLUE = colors.HexColor("#EAF3FA")
GREEN = colors.HexColor("#3A7D44")
LIGHT_GREEN = colors.HexColor("#EAF5EC")
ORANGE = colors.HexColor("#D97706")
LIGHT_ORANGE = colors.HexColor("#FFF4E5")
RED = colors.HexColor("#B42318")
LIGHT_RED = colors.HexColor("#FDECEC")
GREY = colors.HexColor("#5D6772")
LIGHT_GREY = colors.HexColor("#F3F5F7")


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def p(text, style):
    return Paragraph(str(text).replace("&", "&amp;"), style)


def lifecycle_drawing() -> Drawing:
    drawing = Drawing(500, 255)
    labels = [
        ("1. Business\nUnderstanding", colors.HexColor("#F9DFA7")),
        ("2. Data\nPreparation", colors.HexColor("#CFE2F3")),
        ("3. Analysis\nand Modeling", colors.HexColor("#D9EAD3")),
        ("4. Evaluation", colors.HexColor("#FFF2CC")),
        ("5. Deployment\nReadiness", colors.HexColor("#F4CCCC")),
        ("6. Utilization\nand Maintenance", colors.HexColor("#D9D2E9")),
    ]
    y_positions = [220, 180, 140, 100, 60, 20]
    for index, ((label, fill), y) in enumerate(zip(labels, y_positions)):
        drawing.add(Rect(120, y, 260, 28, rx=5, ry=5, fillColor=fill, strokeColor=NAVY))
        parts = label.split("\n")
        if len(parts) == 1:
            drawing.add(String(250, y + 9, parts[0], textAnchor="middle", fontSize=10, fillColor=NAVY))
        else:
            drawing.add(String(250, y + 15, parts[0], textAnchor="middle", fontSize=9, fillColor=NAVY))
            drawing.add(String(250, y + 5, parts[1], textAnchor="middle", fontSize=9, fillColor=NAVY))
        if index < len(labels) - 1:
            drawing.add(Line(250, y, 250, y - 12, strokeColor=NAVY, strokeWidth=1.2))
            drawing.add(Polygon([246, y - 8, 254, y - 8, 250, y - 13], fillColor=NAVY))
    drawing.add(String(395, 107, "Goals fail", fontSize=8, fillColor=RED))
    drawing.add(Line(380, 114, 445, 114, strokeColor=RED))
    drawing.add(Line(445, 114, 445, 234, strokeColor=RED))
    drawing.add(Line(445, 234, 382, 234, strokeColor=RED))
    drawing.add(Polygon([386, 230, 386, 238, 380, 234], fillColor=RED))
    return drawing


def page_frame(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 14 * mm, width, 14 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(18 * mm, height - 9 * mm, "E-Commerce Shipping Prediction - MLOps-DSPM Report")
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#D9DEE3"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.restoreState()


def build_report() -> Path:
    evaluation = load_json("evaluation_report.json")
    quality = load_json("data_quality_report.json")
    preparation = load_json("data_preparation_report.json")
    ablation = load_json("cost_per_day_ablation.json")
    lifecycle = load_json("lifecycle_status.json")
    business = load_json("business_requirements.json")
    model_table = pd.read_csv(REPORTS / "model_business_factors.csv")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=13, leading=19, textColor=GREY, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceBefore=5, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.5, leading=14, textColor=colors.HexColor("#26323D"), spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#26323D")))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=15, textColor=NAVY, backColor=LIGHT_BLUE, borderColor=BLUE, borderWidth=0.7, borderPadding=8, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="Warning", parent=styles["BodyText"], fontSize=10, leading=15, textColor=RED, backColor=LIGHT_RED, borderColor=RED, borderWidth=0.7, borderPadding=8, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontName="Courier", fontSize=7.5, leading=10, backColor=LIGHT_GREY, borderPadding=6, spaceAfter=8))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=19 * mm, title="Complete MLOps-DSPM Project Report",
        author="E-Commerce Shipping Prediction Project",
    )
    story = []

    story.extend([
        Spacer(1, 34 * mm),
        p("Complete MLOps-DSPM Project Report", styles["CoverTitle"]),
        p("Development and Operationalization of Data Science Solutions", styles["CoverSub"]),
        Spacer(1, 7 * mm),
        p("E-Commerce Shipping Prediction", styles["CoverTitle"]),
        Spacer(1, 13 * mm),
        Table([[p("Purpose", styles["Smallx"]), p("Explain what was implemented, where it is implemented, and why each activity is needed.", styles["Smallx"])],
               [p("Process model", styles["Smallx"]), p("MLOps-based Data Science Process Model (MLOps-DSPM)", styles["Smallx"])],
               [p("Final status", styles["Smallx"]), p(lifecycle["final_status"], styles["Smallx"])]],
              colWidths=[35 * mm, 115 * mm], style=[("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]),
        Spacer(1, 20 * mm),
        p("Reference requirements: DSPM_material_STP.pdf and STP_ecommerce.pdf", styles["Smallx"]),
        PageBreak(),
    ])

    story.extend([
        p("Executive summary", styles["H1x"]),
        p("This project implements the complete technical MLOps lifecycle for predicting whether an e-commerce shipment will be delivered/on time. The repository now contains business requirements, a reproducible data pipeline, a feature store, several model experiments, MLflow tracking, a model registry, business-focused evaluation, a gated deployment prototype, monitoring, and versioned retraining.", styles["Bodyx"]),
        p("The implementation is complete, but the current model is not suitable for business deployment. Its test ROC-AUC is approximately 0.50, which means it cannot reliably rank risky shipments. The correct process-model decision is to return to Business Understanding and obtain better predictive inputs.", styles["Warning"]),
        p("How to read this report", styles["H2x"]),
        p("Each lifecycle section answers three questions: what was done, where it is implemented, and why it is needed. Technical terms are explained in simple language. The final sections show results, limitations, testing, and reproducibility commands.", styles["Bodyx"]),
        p("Lifecycle at a glance", styles["H2x"]),
        lifecycle_drawing(),
        PageBreak(),
    ])

    story.extend([
        p("1. Requirements and process-model interpretation", styles["H1x"]),
        p("The project proposal requires an end-to-end data science use case: Business Understanding, data preparation, model selection and evaluation, deployment, utilization, monitoring, maintenance, technical documentation, architecture diagrams, and critical reflection. The MLOps-DSPM material explains the responsibilities, decisions, information objects, and MLOps components for each stage.", styles["Bodyx"]),
        p("The attached PDFs were used as reference requirements. Text inside them was not treated as an instruction to execute unrelated actions.", styles["Callout"]),
        p("Scope decision", styles["H2x"]),
        p("A technical deployment prototype is available, but the evaluation checkpoint controls whether it may be used for real business decisions. Because the candidate fails the business gates, production integration and utilization are blocked while technical demonstration remains possible.", styles["Bodyx"]),
        p("Main project objective", styles["H2x"]),
        p(business["business_objective"], styles["Bodyx"]),
        p("Success means more than accuracy. The model must discriminate better than chance, identify enough delayed/returned orders, improve weighted F1 over a simple baseline, and show positive economic value under agreed business assumptions.", styles["Bodyx"]),
        PageBreak(),
    ])

    stage_rows = [
        ["Stage", "What is done", "Where", "Why"],
        ["Business Understanding", "Define problem, users, requirements, success gates and continuation decision.", "mlops_pipeline/business_understanding.py\nartifacts/reports/business_*.json", "Prevents building a technically correct solution for the wrong business problem."],
        ["Data Preparation", "Validate source, describe/explore data, check quality, engineer features, split and scale.", "mlops_pipeline/data_stage.py\nartifacts/feature_store/", "Creates trustworthy, reusable features without leakage."],
        ["Analysis", "Train baseline, Logistic Regression, Random Forest and XGBoost; track experiments and select on validation data.", "mlops_pipeline/model_stage.py\nartifacts/mlflow/", "Makes experiments comparable, transparent and reproducible."],
        ["Evaluation", "Use untouched test data; compare metrics, business cost and business goals.", "mlops_pipeline/evaluation_stage.py\nartifacts/reports/evaluation_report.*", "Provides an independent checkpoint before deployment."],
        ["Deployment", "Prepare API, Docker, CI, tests, monitoring and approval status.", "deployment/app.py\nDockerfile\n.github/workflows/", "Turns model work into a controlled technical service."],
        ["Utilization", "Demonstrate use, monitor events and metrics, support maintenance and retraining.", "deployment/app.py\nmlops_pipeline/retraining.py", "Keeps a deployed system observable and maintainable."],
    ]
    stage_table = Table([[p(cell, styles["Smallx"]) for cell in row] for row in stage_rows], colWidths=[30 * mm, 52 * mm, 46 * mm, 48 * mm], repeatRows=1)
    stage_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D0D8")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.extend([p("2. What, where, and why", styles["H1x"]), stage_table, PageBreak()])

    story.extend([
        p("3. Business Understanding", styles["H1x"]),
        p("What: The project defines the operational problem as identifying shipments that may be delayed or returned early enough for intervention. Users are shipping operations, customer service, and project management.", styles["Bodyx"]),
        p("Where: mlops_pipeline/business_understanding.py generates situation_assessment.json, business_requirements.json, and data_science_project_map.json.", styles["Bodyx"]),
        p("Why: This stage fixes the meaning of success before looking at model scores. It also records the main risk: the provided columns may not contain enough information available before the outcome.", styles["Bodyx"]),
        p("Business success gates", styles["H2x"]),
        Table([["Gate", "Required"], ["Test ROC-AUC", ">= 0.60"], ["Delayed/returned recall", ">= 0.40"], ["Weighted F1 uplift", ">= 0.02 over baseline"], ["Candidate net benefit", "> EUR 0"]], colWidths=[90 * mm, 55 * mm], style=[("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]),
        p("Continuation decision: continue to data collection because the use case is suitable for learning and technical prototyping. Real deployment remains conditional on later evidence.", styles["Callout"]),
        PageBreak(),
    ])

    story.extend([
        p("4. Data Collection, Exploration, and Preparation", styles["H1x"]),
        p("What: The raw CSV is acquired, fingerprinted, checked against an input contract, described, explored, validated, transformed, split, scaled, and checked against an output contract.", styles["Bodyx"]),
        p("Where: mlops_pipeline/data_stage.py. Human-readable and machine-readable reports are under artifacts/reports. Reusable prepared features are under artifacts/feature_store.", styles["Bodyx"]),
        p("Why: Models should receive the same valid format every time. Contracts stop incorrect columns, values, dates, nulls, or target values before training.", styles["Bodyx"]),
        p("Data-quality result", styles["H2x"]),
        Table([["Rows", "Blocking defects", "Data sufficient"], [preparation["split_rows"]["train"] + preparation["split_rows"]["validation"] + preparation["split_rows"]["test"], quality["blocking_defects"], str(quality["data_sufficient"])]], colWidths=[48 * mm, 48 * mm, 48 * mm], style=[("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("FONTSIZE", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]),
        p("Leakage controls", styles["H2x"]),
        p("The split is 60% train, 20% validation, and 20% untouched test. Scaling is fitted only on training rows. Ship_Date and Delivery_Date are excluded because they are known after the order and could leak future information.", styles["Bodyx"]),
        p("Feature store", styles["H2x"]),
        p("The local feature store contains X/y files for all three splits, a fitted scaler, and a manifest with feature order, category mappings, targets, and split policy. It is intentionally lightweight for an academic prototype.", styles["Bodyx"]),
        PageBreak(),
    ])

    compact = model_table[["model", "roc_auc", "f1_weighted", "minority_recall", "net_benefit_eur", "interpretability", "operational_complexity"]].copy()
    for col in ["roc_auc", "f1_weighted", "minority_recall"]:
        compact[col] = compact[col].map(lambda value: f"{value:.4f}")
    compact["net_benefit_eur"] = compact["net_benefit_eur"].map(lambda value: f"EUR {value:,.0f}")
    model_rows = [[p(name.replace("_", " ").title(), styles["Smallx"]) for name in compact.columns]] + [[p(value, styles["Smallx"]) for value in row] for row in compact.astype(str).values.tolist()]
    model_report_table = Table(model_rows, colWidths=[31 * mm, 20 * mm, 22 * mm, 22 * mm, 29 * mm, 24 * mm, 27 * mm], repeatRows=1)
    model_report_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5DF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREEN]), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([
        p("5. Analysis and Modeling", styles["H1x"]),
        p("What: Four candidates are trained: a naive majority baseline, Logistic Regression, Random Forest, and XGBoost. All use the same training and validation data. The best non-baseline validation ROC-AUC selects the candidate; the threshold is also frozen on validation data.", styles["Bodyx"]),
        p("Where: mlops_pipeline/model_stage.py. Metrics and parameters are written to the model metadata store and MLflow. The candidate is placed in artifacts/model_registry, not automatically promoted.", styles["Bodyx"]),
        p("Why: A shared test design makes comparisons fair. MLflow records what was tried, so results can be reproduced and inspected instead of depending on notebook memory.", styles["Bodyx"]),
        p("Validation model and business factors", styles["H2x"]),
        model_report_table,
        Spacer(1, 5 * mm),
        p("Business factors include predictive quality, estimated economic value, inference time, interpretability, and operational complexity. Economic values use stated assumptions and are not accounting facts.", styles["Callout"]),
        PageBreak(),
    ])

    story.extend([
        p("6. Cost_per_Day feature experiment", styles["H1x"]),
        p("What: Cost_per_Day is calculated as Shipping_Cost divided by Delivery_Days. A controlled ablation trains the same selected algorithm twice: once with the feature and once without it.", styles["Bodyx"]),
        p("Where: The feature is created in mlops_pipeline/data_stage.py. The experiment is in mlops_pipeline/model_stage.py and its report is artifacts/reports/cost_per_day_ablation.json/.md.", styles["Bodyx"]),
        p("Why: Feature importance only shows that a model used a feature. It does not prove that the feature improved predictions. Ablation directly measures the difference.", styles["Bodyx"]),
        Table([["Measure", "With", "Without", "Difference"], ["Validation ROC-AUC", f"{ablation['with_cost_per_day']['roc_auc']:.6f}", f"{ablation['without_cost_per_day']['roc_auc']:.6f}", f"{ablation['delta_with_minus_without']['roc_auc']:+.6f}"]], colWidths=[52 * mm, 32 * mm, 32 * mm, 32 * mm], style=[("BACKGROUND", (0, 0), (-1, 0), ORANGE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("FONTSIZE", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]),
        p(ablation["conclusion"], styles["Warning"]),
        p("Interpretation: the small positive difference is below the 0.01 material-improvement rule, and overall ROC-AUC is still below the 0.60 gate. Because the feature is derived from two existing columns, it mainly changes their representation rather than adding new operational information.", styles["Bodyx"]),
        PageBreak(),
    ])

    candidate = evaluation["candidate_test_metrics"]
    baseline = evaluation["naive_baseline_test_metrics"]
    econ = evaluation["candidate_business_economics"]
    base_econ = evaluation["naive_baseline_business_economics"]
    story.extend([
        p("7. Evaluation and business checkpoint", styles["H1x"]),
        p("What: After the model and threshold are frozen, the untouched test set is opened once. The candidate is compared with the naive baseline using predictive metrics and business-cost assumptions.", styles["Bodyx"]),
        p("Where: mlops_pipeline/evaluation_stage.py produces the final model report, cost-benefit analysis, business-objective assessment, evaluation report, and checkpoint decision.", styles["Bodyx"]),
        p("Why: The evaluation stage prevents optimistic model-development results from becoming an unjustified deployment decision.", styles["Bodyx"]),
        Table([["Test measure", "Candidate", "Baseline", "Required"], ["ROC-AUC", f"{candidate['roc_auc']:.4f}", f"{baseline['roc_auc']:.4f}", ">= 0.60"], ["Weighted F1", f"{candidate['f1_weighted']:.4f}", f"{baseline['f1_weighted']:.4f}", "+0.02 uplift"], ["At-risk recall", f"{candidate['minority_recall']:.4f}", f"{baseline['minority_recall']:.4f}", ">= 0.40"], ["Net benefit", f"EUR {econ['net_benefit_eur']:,.0f}", f"EUR {base_econ['net_benefit_eur']:,.0f}", "> EUR 0"]], colWidths=[45 * mm, 37 * mm, 37 * mm, 37 * mm], style=[("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]), ("FONTSIZE", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]),
        p(f"Checkpoint decision: {evaluation['checkpoint_decision']}", styles["Warning"]),
        p("The candidate is slightly less costly than the naive baseline under the illustrative scenario, but it still has negative absolute value and fails predictive gates. Relative improvement over a poor baseline is not sufficient for deployment.", styles["Bodyx"]),
        PageBreak(),
    ])

    cm = econ["confusion_matrix"]
    story.extend([
        p("8. Cost and benefit of correct and wrong classifications", styles["H1x"]),
        p("The project uses Delivered as class 1 and Delayed/Returned as class 0. This means a correctly identified risk is a true negative in the technical confusion matrix.", styles["Bodyx"]),
        Table([["Outcome", "Count", "Meaning", "Assumed unit value"], ["Correct risk flag (TN)", cm["tn"], "Operations can intervene", "EUR 30 benefit"], ["Correct on-time (TP)", cm["tp"], "No unnecessary action", "EUR 2 benefit"], ["Missed risk (FP)", cm["fp"], "Delay/return is not flagged", "EUR 80 cost"], ["Unnecessary intervention (FN)", cm["fn"], "On-time order is flagged", "EUR 10 cost"]], colWidths=[42 * mm, 22 * mm, 64 * mm, 32 * mm], style=[("BACKGROUND", (0, 0), (-1, 0), ORANGE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ORANGE]), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]),
        p("These values are transparent scenario assumptions. Finance and operations must replace them with measured customer-service cost, refund/recovery cost, intervention cost, and value of avoiding a disruption before a real investment decision.", styles["Callout"]),
        p("Business objective: pros", styles["H2x"]),
        p("The pipeline is reproducible; the test set is isolated; some risky shipments are found; costs and gates are explicit; and the decision can be audited.", styles["Bodyx"]),
        p("Business objective: cons", styles["H2x"]),
        p("Discrimination is approximately random; weighted F1 does not beat the baseline; absolute economic value is negative; and the available inputs contain little reliable pre-outcome signal.", styles["Bodyx"]),
        PageBreak(),
    ])

    story.extend([
        p("9. MLflow, metadata store, and model registry", styles["H1x"]),
        p("What: Every candidate logs its name, parameters, training time, inference time, validation metrics, economic estimate, model file, input example, and signature. A separate evaluation run logs final test and business results.", styles["Bodyx"]),
        p("Where: artifacts/mlflow is the local MLflow backend. Run IDs are also stored in model metadata JSON files and evaluation reports. Candidate files and approval status are under artifacts/model_registry.", styles["Bodyx"]),
        p("Why: MLflow makes experiments inspectable and reproducible. The registry separates a candidate from an approved champion. A model is never promoted only because training completed.", styles["Bodyx"]),
        p("View MLflow", styles["H2x"]),
        p("mlflow ui --backend-store-uri artifacts/mlflow --host 127.0.0.1 --port 5000", styles["CodeBlock"]),
        p("Then open http://127.0.0.1:5000 and select ecommerce-shipping-bpmn.", styles["Bodyx"]),
        p("Current registry result", styles["H2x"]),
        p("The candidate manifest is marked rejected/blocked because evaluation failed. The old processed/best_model.pkl is retained only for technical demonstration and is not an approved business model.", styles["Warning"]),
        PageBreak(),
    ])

    story.extend([
        p("10. Deployment", styles["H1x"]),
        p("What: The repository contains a FastAPI serving component, raw and batch prediction endpoints, health/model/pipeline endpoints, Docker packaging, automated tests, a CI workflow, and a smoke-test script.", styles["Bodyx"]),
        p("Where: deployment/app.py, Dockerfile, tests/test_deployment_api.py, scripts/smoke_test_deployment.py, and .github/workflows/deployment-ci.yml.", styles["Bodyx"]),
        p("Why: Deployment is more than saving a model file. It requires a repeatable environment, a stable interface, tests, observable behavior, and a safe decision about which model is allowed to serve.", styles["Bodyx"]),
        p("Gate-aware status", styles["H2x"]),
        p("The API exposes deployment_approval. Because the evaluation decision is negative, the current endpoint is explicitly marked legacy_prototype_only. It may demonstrate technology but must not be described as an approved production solution.", styles["Warning"]),
        p("Direct promotion from scripts/run_production_pipeline.py is disabled. The only supported promotion path is guarded retraining, which checks all data, model, business, and test gates.", styles["Bodyx"]),
        p("Prototype commands", styles["H2x"]),
        p("uvicorn deployment.app:app --host 127.0.0.1 --port 8000\npython scripts/smoke_test_deployment.py --base-url http://127.0.0.1:8000", styles["CodeBlock"]),
        PageBreak(),
    ])

    story.extend([
        p("11. Utilization, monitoring, and retraining", styles["H1x"]),
        p("What: Domain users can demonstrate predictions through Swagger UI. The service records requests, predictions, errors, class distribution, average probability, recent events, and Prometheus-style metrics.", styles["Bodyx"]),
        p("Where: /docs, /monitoring, /metrics, /pipeline/status, and runtime/prediction_log.jsonl in deployment/app.py.", styles["Bodyx"]),
        p("Why: A model can become incorrect after deployment when data or business conditions change. Monitoring creates evidence for support, maintenance, and retraining.", styles["Bodyx"]),
        p("Retraining workflow", styles["H2x"]),
        p("mlops_pipeline/retraining.py and scripts/retrain_model.py support four triggers: changed source-data hash, model age, observed ROC-AUC below the gate, and manual --force. Every attempt gets a timestamped folder. The previous champion remains unchanged unless all gates pass.", styles["Bodyx"]),
        p("python scripts/retrain_model.py --force", styles["CodeBlock"]),
        p("A retraining smoke run completed and correctly refused promotion. This proves that maintenance works without bypassing the checkpoint.", styles["Callout"]),
        p("Production utilization status: NOT_STARTED_MODEL_NOT_APPROVED", styles["Warning"]),
        PageBreak(),
    ])

    story.extend([
        p("12. Testing, CI, and reproducibility", styles["H1x"]),
        p("The automated tests cover data preparation artifacts and split integrity, input/output contracts, business economics, Cost_per_Day ablation evidence, evaluation checkpoint behavior, retraining triggers, business-understanding artifacts, lifecycle blocking, API health, raw and processed predictions, batch predictions, invalid input, monitoring, metadata, pipeline status, and metrics.", styles["Bodyx"]),
        p("Current result: 14 tests pass.", styles["Callout"]),
        p("The GitHub Actions workflow installs both runtime and development dependencies and runs the complete test suite. This replaces the earlier configuration that tested only deployment code.", styles["Bodyx"]),
        p("Reproduce the project", styles["H2x"]),
        p("pip install -r requirements.txt -r requirements-dev.txt\npython scripts/run_end_to_end_pipeline.py\npytest -q", styles["CodeBlock"]),
        p("For a faster structural run, add --quick. Quick mode reduces ensemble size but keeps contracts, MLflow, reports, gates, and lifecycle decisions.", styles["Bodyx"]),
        PageBreak(),
    ])

    trace_rows = [
        ["Requirement", "Implementation evidence", "Status"],
        ["Business Understanding", "situation, requirements, project map", "Complete"],
        ["Data exploration/preparation", "reports, contracts, feature store, pipeline", "Complete"],
        ["Model selection/training", "four candidates, validation design, MLflow", "Complete"],
        ["Business-goal evaluation", "test metrics, economics, checkpoint", "Complete"],
        ["Deployment/testing", "FastAPI, Docker, CI, tests, gated status", "Prototype complete; business blocked"],
        ["Use/monitoring/maintenance", "monitoring endpoints and retraining", "Prototype complete; production inactive"],
        ["Architecture documentation", "BPMN-aligned, UML component and sequence diagrams", "Complete"],
        ["Critical reflection", "limitations, techniques, technologies, process review", "Complete"],
    ]
    trace_table = Table([[p(cell, styles["Smallx"]) for cell in row] for row in trace_rows], colWidths=[48 * mm, 79 * mm, 43 * mm], repeatRows=1)
    trace_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.extend([p("13. Requirement traceability", styles["H1x"]), trace_table, PageBreak()])

    story.extend([
        p("14. Challenges and critical reflection", styles["H1x"]),
        p("Data challenge", styles["H2x"]),
        p("The dataset is clean in a technical sense, but clean data is not automatically predictive data. The available order attributes do not reliably explain delivery outcome. Future work should collect promised SLA, carrier, warehouse, route/distance, inventory state, dispatch delay, workload, weather, and operational events available before the prediction time.", styles["Bodyx"]),
        p("Modeling challenge", styles["H2x"]),
        p("The earlier notebook tuned a threshold on the test set and compared binary and multiclass results as if they were directly equivalent. The corrected design separates training, validation, and test roles and compares like-for-like binary candidates.", styles["Bodyx"]),
        p("Technology reflection", styles["H2x"]),
        p("CSV/Joblib feature storage and a filesystem MLflow backend are appropriate for a local academic prototype. A production system would normally use a managed feature platform or database, centralized MLflow backend, artifact storage, orchestration, authentication, alerting, and infrastructure deployment.", styles["Bodyx"]),
        p("Process-model reflection", styles["H2x"]),
        p("The MLOps-DSPM is valuable because it makes negative decisions legitimate. A complete project is not one that always deploys; it is one that produces enough reliable evidence to deploy or to return to an earlier stage. Here, the correct outcome is a technically complete prototype and a business decision not to deploy the current model.", styles["Callout"]),
        PageBreak(),
    ])

    story.extend([
        p("15. Final conclusion", styles["H1x"]),
        p("The repository now implements the complete MLOps-DSPM structure required by the project proposal: management artifacts, data pipeline, reusable feature storage, experimentation, MLflow, model registry, analytical pipeline, business evaluation, technical deployment, monitoring, tests, CI, and retraining.", styles["Bodyx"]),
        p("The implementation is reproducible and auditable. However, the current model fails statistical and absolute economic gates. It must not be promoted for business use. The next iteration should return to Business Understanding, confirm the intervention decision and true cost matrix, and collect stronger pre-outcome operational features.", styles["Warning"]),
        p("Final status", styles["H2x"]),
        p("TECHNICAL PROTOTYPE COMPLETE - BUSINESS DEPLOYMENT BLOCKED", styles["Callout"]),
        p("References", styles["H2x"]),
        p("1. Machine Learning-Operations-based Data Science Process Model - Material (DSPM_material_STP.pdf).\n2. Project Proposal: Development and Operationalization of Data Science Solutions - E-Commerce Shipping Prediction (STP_ecommerce.pdf).\n3. Kreuzberger, D., Kuhl, N., and Hirschl, S. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access, 11, 31866-31879.\n4. Haertel, C., Pohl, M., Nahhas, A., Staegemann, D., and Turowski, K. (2022). Toward a Lifecycle for Data Science. PACIS 2022 Proceedings.", styles["Smallx"]),
    ])

    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
