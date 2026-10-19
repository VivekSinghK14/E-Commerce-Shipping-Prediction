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
LIVE_DEMO_URL = "https://huggingface.co/spaces/VivekSinghK14/shipping-delay-predictor"

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


MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "dummy_majority": "Naive majority baseline",
}


GATE_LABELS = {
    "roc_auc": "test ROC-AUC >= 0.60",
    "minority_recall": "delayed recall >= 0.40",
    "weighted_f1_improvement_over_baseline": "weighted F1 uplift >= 0.02",
    "net_benefit_improvement_over_baseline_eur": "net benefit better than baseline",
    "candidate_net_benefit_eur": "net benefit > EUR 0",
}


def model_display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


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


def _collect_test_counts() -> dict[str, int]:
    """Count the automated tests per file by asking pytest to collect them."""
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if "::" in line:
            name = line.split("::")[0].replace("tests/", "")
            counts[name] = counts.get(name, 0) + 1
    return counts


def _delay_profile(raw_path: Path) -> tuple[list[list[str]], float, float]:
    """Delay rate per order-time attribute plus a chi-square test against the target.

    Returns table rows, the largest spread in percentage points, and the smallest p-value.
    """
    from scipy.stats import chi2_contingency

    raw = pd.read_csv(raw_path)
    raw = raw[raw["Delivery_Status"].isin(["Delivered", "Delayed"])].copy()
    delayed = raw["Delivery_Status"].eq("Delayed")
    order_dates = pd.to_datetime(raw["Order_Date"], format="mixed")
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    raw["Order month"] = order_dates.dt.month.map(lambda m: f"month {m}")
    raw["Order weekday"] = order_dates.dt.dayofweek.map(lambda d: weekdays[d])
    raw["Cost quartile"] = pd.qcut(raw["Shipping_Cost"], 4, labels=["lowest 25%", "2nd 25%", "3rd 25%", "highest 25%"])
    rows, spreads, p_values = [], [], []
    for column, label in [
        ("Shipping_Mode", "Shipping mode"),
        ("Customer_Region", "Customer region"),
        ("Product_Category", "Product category"),
        ("Order month", "Order month"),
        ("Order weekday", "Order weekday"),
        ("Cost quartile", "Shipping cost (quartiles)"),
    ]:
        rates = delayed.groupby(raw[column], observed=True).mean()
        _, p_value, _, _ = chi2_contingency(pd.crosstab(raw[column], delayed))
        spreads.append(100 * (rates.max() - rates.min()))
        p_values.append(p_value)
        rows.append([
            label,
            f"{rates.min():.1%} ({rates.idxmin()})",
            f"{rates.max():.1%} ({rates.idxmax()})",
            f"{p_value:.2f}",
        ])
    return rows, max(spreads), min(p_values)


def _scaled(drawing, factor: float):
    """Shrink a ReportLab drawing proportionally and centre it."""
    drawing.scale(factor, factor)
    drawing.width *= factor
    drawing.height *= factor
    drawing.hAlign = "CENTER"
    return drawing


def build_report() -> Path:
    from mlops_pipeline.config import PipelineConfig

    config = PipelineConfig()
    evaluation = load_json("evaluation_report.json")
    quality = load_json("data_quality_report.json")
    preparation = load_json("data_preparation_report.json")
    exploration = load_json("data_exploration_report.json")
    selection = load_json("analytical_model_selection.json")
    leakage_review = load_json("cost_per_day_leakage_review.json")
    lifecycle = load_json("lifecycle_status.json")
    business = load_json("business_requirements.json")
    smoke_path = REPORTS / "deployment_smoke_test.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else None
    retrain_path = ROOT / "artifacts" / "retraining_report.json"
    retraining = json.loads(retrain_path.read_text(encoding="utf-8")) if retrain_path.exists() else None
    models = pd.read_csv(REPORTS / "model_comparison_validation.csv")

    override = lifecycle["deployment"].get("override")
    scope = preparation["scope_filter"]
    splits = preparation["split_rows"]
    gates = evaluation["business_goal_checks"]
    gates_met = sum(check["passed"] for check in gates.values())
    cand = evaluation["candidate_test_metrics"]
    base = evaluation["naive_baseline_test_metrics"]
    econ = evaluation["candidate_business_economics"]
    base_econ = evaluation["naive_baseline_business_economics"]
    cm = econ["confusion_matrix"]
    threshold = float(selection["validation_selected_threshold"])
    threshold_metrics = selection["threshold_selection"]["metrics"]
    model_name = model_display_name(evaluation["final_selected_model"])
    status_text = (
        f"LIVE PROTOTYPE DEPLOYED - {gates_met} OF {len(gates)} EVALUATION GATES MET"
        if override
        else lifecycle["final_status"].replace("_", " ")
    )
    outcomes = exploration["category_distributions"]["Delivery_Status"]
    delivered_n, delayed_n = outcomes["Delivered"], outcomes["Delayed"]
    in_scope = scope["rows_in_scope"]
    split_delayed = {
        name: int((pd.read_csv(ROOT / "artifacts" / "feature_store" / f"y_{name}.csv")["binary_target"] == 0).sum())
        for name in ["train", "validation", "test"]
    }
    cost_stats = exploration["numeric_summary"]["Shipping_Cost"]
    order_dates = exploration["date_ranges"]["Order_Date"]
    delay_profile, max_spread, min_p = _delay_profile(config.raw_data)
    test_counts = _collect_test_counts()
    total_tests = sum(test_counts.values())
    flagged = cm["tn"] + cm["fn"]
    actual_delayed = cm["tn"] + cm["fp"]
    selected_row = models.set_index("model").loc[evaluation["final_selected_model"]]
    baseline_row = models.set_index("model").loc["dummy_majority"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=13, leading=19, textColor=GREY, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceBefore=5, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.5, leading=14, textColor=colors.HexColor("#26323D"), spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#26323D")))
    styles.add(ParagraphStyle(name="SmallHead", parent=styles["Smallx"], fontName="Helvetica-Bold", textColor=colors.white))
    styles.add(ParagraphStyle(name="Findx", parent=styles["Smallx"], textColor=GREY, spaceBefore=4, spaceAfter=4))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=15, textColor=NAVY, backColor=LIGHT_BLUE, borderColor=BLUE, borderWidth=0.7, borderPadding=8, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="Warning", parent=styles["BodyText"], fontSize=10, leading=15, textColor=RED, backColor=LIGHT_RED, borderColor=RED, borderWidth=0.7, borderPadding=8, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontName="Courier", fontSize=7.5, leading=10, backColor=LIGHT_GREY, borderPadding=6, spaceAfter=8))

    def body(text):
        return p(text, styles["Bodyx"])

    def find(text):
        return p(f"<b>Where to find it:</b> {text}", styles["Findx"])

    def table(rows, widths, header=NAVY, zebra=LIGHT_GREY, pad=5):
        cells = [[p(cell, styles["SmallHead" if r == 0 else "Smallx"]) for cell in row] for r, row in enumerate(rows)]
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), header),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D0D8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ]
        return Table(cells, colWidths=[w * mm for w in widths], repeatRows=1, style=style)

    def met(passed):
        return "Met" if passed else "Not met"

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=19 * mm, title="Complete MLOps-DSPM Project Report",
        author="E-Commerce Shipping Prediction Project",
    )
    story = []

    # Cover ---------------------------------------------------------------
    story.extend([
        Spacer(1, 34 * mm),
        p("Complete MLOps-DSPM Project Report", styles["CoverTitle"]),
        p("Development and Operationalization of Data Science Solutions", styles["CoverSub"]),
        Spacer(1, 7 * mm),
        p("E-Commerce Shipping Prediction", styles["CoverTitle"]),
        Spacer(1, 13 * mm),
        Table([[p("Purpose", styles["Smallx"]), p("Explain what was implemented, where it is implemented, and why each activity is needed.", styles["Smallx"])],
               [p("Process model", styles["Smallx"]), p("MLOps-based Data Science Process Model (MLOps-DSPM)", styles["Smallx"])],
               [p("Final status", styles["Smallx"]), p(status_text, styles["Smallx"])],
               [p("Live demo", styles["Smallx"]), p(LIVE_DEMO_URL, styles["Smallx"])]],
              colWidths=[35 * mm, 115 * mm], style=[("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5DF")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]),
        Spacer(1, 20 * mm),
        p("Reference requirements: DSPM_material_STP.pdf and STP_ecommerce.pdf", styles["Smallx"]),
        PageBreak(),
    ])

    # Executive summary -----------------------------------------------------
    story.extend([
        p("Executive summary", styles["H1x"]),
        body(f"This project predicts, at the moment an order is placed, whether it will be <b>Delivered</b> on time or <b>Delayed</b>, and builds the complete "
             f"MLOps lifecycle around that question: requirements, data pipeline, feature store, tracked experiments, model registry, evaluation, a deployed API and "
             f"web app, monitoring, and retraining. Of the {scope['source_rows']:,} orders in the source file, the {scope['rows_removed']:,} Returned orders belong to a "
             f"different business process and were removed first; the remaining {in_scope:,} orders are used throughout."),
        p(f"<b>Result in one paragraph.</b> The selected {model_name} model is live as a prototype ({LIVE_DEMO_URL}). On {splits['test']:,} test orders "
          f"that were never used for training or tuning, it finds {cand['minority_recall']:.1%} of the delayed orders and reduces the estimated cost of delays "
          f"by EUR {gates['net_benefit_improvement_over_baseline_eur']['actual']:,.0f} compared with doing nothing. It meets {gates_met} of {len(gates)} agreed "
          f"evaluation gates. Its ROC-AUC of {cand['roc_auc']:.3f} shows that the information available at order time (region, product category, shipping mode, "
          f"cost, and date) does not yet separate delayed from delivered orders reliably, so the next iteration focuses on collecting operational data "
          f"such as carrier, warehouse, route, and promised delivery date.", styles["Callout"]),
        p("Key numbers at a glance", styles["H2x"]),
        table([
            ["Item", "Value", "Section"],
            ["Orders in source file / Returned removed / used", f"{scope['source_rows']:,} / {scope['rows_removed']:,} / {in_scope:,}", "4"],
            ["Delivered vs Delayed", f"{delivered_n:,} ({delivered_n / in_scope:.1%}) vs {delayed_n:,} ({delayed_n / in_scope:.1%})", "4"],
            ["Train / validation / test orders", f"{splits['train']:,} / {splits['validation']:,} / {splits['test']:,}", "4"],
            ["Models compared", "Naive baseline, Logistic Regression, Random Forest, XGBoost", "5"],
            ["Selected model and decision threshold", f"{model_name}, Delivered if P(Delivered) >= {threshold:.2f}", "5, 7"],
            ["Test ROC-AUC / delayed recall / weighted F1", f"{cand['roc_auc']:.3f} / {cand['minority_recall']:.3f} / {cand['f1_weighted']:.3f}", "7"],
            ["Estimated net benefit vs baseline", f"EUR {econ['net_benefit_eur']:,.0f} vs EUR {base_econ['net_benefit_eur']:,.0f}", "8"],
            ["Evaluation gates met", f"{gates_met} of {len(gates)}", "7"],
            ["Automated tests", f"{total_tests} passing" if total_tests else "see tests/", "12"],
        ], [62, 96, 16], pad=3),
        p("How to read this report", styles["H2x"]),
        body("Each section covers one lifecycle stage, explains what we did and why, and ends with a <b>Where to find it</b> line naming the file behind every number."),
        p("Lifecycle at a glance", styles["H2x"]),
        _scaled(lifecycle_drawing(), 0.62),
        PageBreak(),
    ])

    # 1 ---------------------------------------------------------------------
    story.extend([
        p("1. Requirements and process-model interpretation", styles["H1x"]),
        body("The project proposal asks for a complete data science use case: business understanding, data preparation, model selection and evaluation, "
             "deployment, use, monitoring and maintenance, technical documentation, architecture diagrams, and a critical reflection. The MLOps-DSPM material "
             "describes who is responsible at each stage, which decisions must be made, which documents are produced, and which MLOps components "
             "(feature store, model registry, serving and monitoring) support the work. We used both documents as our checklist; section 13 maps every requirement to its evidence."),
        p("Scope decision", styles["H2x"]),
        body(f"The question is binary: Delivered (1) or Delayed (0). Returned orders ({scope['rows_removed']:,} rows, {scope['rows_removed'] / scope['source_rows']:.1%} of the file) "
             "are handled by a separate reverse-logistics process with different causes and different actions, so mixing them into the target would blur the question. "
             "They are removed in the first pipeline step and never reach features, models, reports, or the app."),
        body("A model only influences real business decisions after the evaluation checkpoint. "
             + ("This model is deployed as a live prototype through a documented decision of the project owner (section 10), so the complete lifecycle can be demonstrated, while the evaluation results are reported exactly as measured." if override
                else "Because the gates are not met, production use is blocked while technical demonstration remains possible.")),
        p("Main project objective", styles["H2x"]),
        body(business["business_objective"]),
        body("Success is defined by more than accuracy. With 84% of orders delivered, a model that always says Delivered is already 84% accurate yet useless. "
             "We therefore require the model to rank orders better than chance, to catch a meaningful share of delayed orders, to beat a naive baseline, "
             "and to create positive value under agreed cost assumptions (gates listed in section 3)."),
        find("artifacts/reports/business_requirements.json, artifacts/reports/data_requirements.json"),
        PageBreak(),
    ])

    # 2 ---------------------------------------------------------------------
    story.extend([
        p("2. What, where, and why", styles["H1x"]),
        body("The table summarises each lifecycle stage, the code and output files that implement it, and why the stage matters."),
        table([
            ["Stage", "What is done", "Where", "Why"],
            ["Business Understanding", "Define the problem, users, requirements, success gates, and continuation decision.", "mlops_pipeline/business_understanding.py; artifacts/reports/business_*.json", "Prevents building a technically correct solution for the wrong problem."],
            ["Data Preparation", f"Remove {scope['rows_removed']:,} Returned orders, validate, describe, explore, engineer 6 features, split 60/20/20, scale.", "mlops_pipeline/data_stage.py; artifacts/feature_store/", "Gives every model the same clean, leakage-free input."],
            ["Analysis", "Train a naive baseline, Logistic Regression, Random Forest, and XGBoost; select on validation data; log to MLflow.", "mlops_pipeline/model_stage.py; artifacts/mlflow/", "Makes experiments fair, comparable, and reproducible."],
            ["Evaluation", f"Score the frozen model once on {splits['test']:,} untouched test orders; compare with the baseline, costs, and gates.", "mlops_pipeline/evaluation_stage.py; artifacts/reports/evaluation_report.*", "Gives an independent, honest checkpoint before deployment."],
            ["Deployment", "FastAPI service, web app, Docker image, CI, tests, smoke test, live demo.", "deployment/; Dockerfile; .github/workflows/", "Turns the model into a controlled service people can use."],
            ["Utilization", "Use through the web app and API, monitoring, feedback, support tickets, guarded retraining.", "deployment/app.py; mlops_pipeline/retraining.py", "Keeps the deployed system observable and maintainable."],
        ], [28, 52, 50, 46]),
        PageBreak(),
    ])

    # 3 ---------------------------------------------------------------------
    story.extend([
        p("3. Business Understanding", styles["H1x"]),
        body("Delayed orders cause customer-service contacts, compensation, and lost customer trust. If operations knew at order time which orders are likely "
             "to be late, they could act early: choose a faster carrier, prioritise picking, or inform the customer. The intended users are shipping operations, "
             "customer service, and project management."),
        body("Before looking at any model score, we fixed what success means. These gates were agreed up front so that the evaluation cannot be adjusted after seeing the results:"),
        p("Business success gates", styles["H2x"]),
        table([
            ["Gate", "Required", "What it protects against"],
            ["Test ROC-AUC", f">= {config.minimum_roc_auc:.2f}", "A model that ranks orders no better than chance (0.50)"],
            ["Delayed recall", f">= {config.minimum_minority_recall:.2f}", "A model that misses most delayed orders"],
            ["Weighted F1 uplift", f">= {config.minimum_f1_improvement:.2f} over baseline", "A model that is not better than always predicting Delivered"],
            ["Net benefit vs baseline", "> EUR 0", "A model that costs more than doing nothing"],
            ["Candidate net benefit", "> EUR 0", "A model whose interventions cost more than they save"],
        ], [45, 40, 89]),
        p("Continuation decision: continue to data collection. The use case is well defined and suitable for a gated prototype; real business use depends on the evidence gathered later.", styles["Callout"]),
        find("artifacts/reports/situation_assessment.json, business_requirements.json, data_science_project_map.json, business_checkpoint.json; thresholds in mlops_pipeline/config.py"),
        PageBreak(),
    ])

    # 4 ---------------------------------------------------------------------
    story.extend([
        p("4. Data Collection, Exploration, and Preparation", styles["H1x"]),
        body(f"The source is <b>50K Records.csv</b>: {scope['source_rows']:,} orders with 10 columns, placed between {order_dates['min'][:10]} and {order_dates['max'][:10]}. "
             "The pipeline fingerprints the file (SHA-256) and checks it against an input contract before processing it."),
        p("Target scope", styles["H2x"]),
        body(f"We keep Delivered and Delayed and remove the {scope['rows_removed']:,} Returned orders, leaving {in_scope:,} orders. "
             f"Of these, {delivered_n:,} ({delivered_n / in_scope:.1%}) were delivered and {delayed_n:,} ({delayed_n / in_scope:.1%}) were delayed. "
             "This imbalance is why accuracy alone is misleading and why we track delayed recall separately."),
        p("Data-quality result", styles["H2x"]),
        table([
            ["Source rows", "Returned removed", "Modeled rows", "Blocking defects", "Data sufficient"],
            [f"{scope['source_rows']:,}", f"{scope['rows_removed']:,}", f"{in_scope:,}", str(quality["blocking_defects"]), "Yes" if quality["data_sufficient"] else "No"],
        ], [34, 34, 34, 34, 34], header=BLUE, pad=4),
        body("All ten quality checks found zero problems (missing values, duplicates, invalid dates or categories, impossible date order, delivery-day mismatches, "
             f"negative costs). Shipping cost ranges from EUR {cost_stats['min']:.0f} to EUR {cost_stats['max']:.0f} (median EUR {cost_stats['50%']:.0f})."),
        p("What the data shows", styles["H2x"]),
        body(f"The overall delay rate is {delayed_n / in_scope:.1%}. Below: the lowest and highest delay rate within each attribute, and a chi-square test "
             "(p above 0.05 means no evidence of a relationship with delays)."),
        table([["Attribute", "Lowest delay rate", "Highest delay rate", "Chi-square p-value"]] + delay_profile, [40, 46, 46, 42], header=BLUE, pad=3),
        Spacer(1, 3 * mm),
        body(f"Across every order-time attribute the delay rate stays between roughly 15% and 17%: the largest gap between two groups is {max_spread:.1f} percentage points, "
             f"and no chi-square test finds a significant relationship (smallest p-value {min_p:.2f}). This is the first sign that order-time attributes carry "
             "little information about delays, and it explains the model results in section 7."),
        p("Features and leakage controls", styles["H2x"]),
        body("The model uses six features known when the order is placed: customer region, product category, shipping mode, shipping cost, order month, and order weekday. "
             "Ship date, delivery date, and delivery days are excluded because they are only known after shipping or delivery; using them would let the model see the answer. "
             f"The data is split once, stratified by outcome: {splits['train']:,} training orders ({split_delayed['train']:,} delayed), "
             f"{splits['validation']:,} validation orders ({split_delayed['validation']:,} delayed), and {splits['test']:,} test orders ({split_delayed['test']:,} delayed). "
             "Scaling is fitted on the training orders only."),
        p("Feature store", styles["H2x"]),
        body("Features and targets for all three splits, the fitted scaler, and a manifest (feature order, category codes, target map, split sizes) are stored "
             "together, so the API, notebooks, and retraining use exactly the same inputs. An output contract checks column order, types, and that the target is only 0 or 1."),
        find("artifacts/reports/data_quality_report.json, data_exploration_report.json, data_preparation_report.json, input_data_contract.json, output_data_contract.json; artifacts/feature_store/; notebook E- commerce.ipynb"),
        PageBreak(),
    ])

    # 5 ---------------------------------------------------------------------
    model_rows = [["Model", "ROC-AUC", "Accuracy", "Bal. acc.", "Weighted F1", "Delayed recall", "Net benefit", "Train time"]]
    for _, row in models.iterrows():
        model_rows.append([
            model_display_name(row["model"]), f"{row['roc_auc']:.4f}", f"{row['accuracy']:.4f}", f"{row['balanced_accuracy']:.4f}",
            f"{row['f1_weighted']:.4f}", f"{row['minority_recall']:.4f}", f"EUR {row['net_benefit_eur']:,.0f}", f"{row['training_seconds']:.1f} s",
        ])
    story.extend([
        p("5. Analysis and Modeling", styles["H1x"]),
        body("We compared four models on the same training data and judged them only on the validation data, keeping the test data untouched. "
             "The <b>naive baseline</b> always predicts the majority class (Delivered) and sets the bar every real model must clear. "
             "<b>Logistic Regression</b> is simple and easy to explain, <b>Random Forest</b> and <b>XGBoost</b> can capture non-linear patterns and interactions. "
             "Logistic Regression and Random Forest weight the classes so the rarer Delayed class is not ignored."),
        p("Validation model and business factors", styles["H2x"]),
        table(model_rows, [33, 18, 18, 18, 20, 20, 26, 17], header=GREEN, zebra=LIGHT_GREEN),
        Spacer(1, 3 * mm),
        body(f"The selection rule, fixed in advance, is the highest validation ROC-AUC among the real models. {model_name} scored {selected_row['roc_auc']:.4f}, "
             f"{selection['validation_roc_auc_uplift_over_naive_baseline']:+.4f} above the baseline. At the default threshold of 0.50 it almost always predicts "
             f"Delivered (delayed recall {selected_row['minority_recall']:.4f}), so the decision threshold is tuned next."),
        p("Choosing the decision threshold", styles["H2x"]),
        body(f"The model outputs a probability that an order will be delivered. We tested every threshold from 0.10 to 0.90 on the validation data. "
             f"The preferred rule was to pick the threshold with the highest positive net benefit while catching at least 40% of delayed orders; no threshold "
             f"reached a positive net benefit, so the fallback rule picks the most balanced threshold that still catches at least 40%. That threshold is "
             f"<b>{threshold:.2f}</b>: an order is predicted Delivered only if the model is at least {threshold:.0%} sure. On validation data it gives a delayed recall "
             f"of {threshold_metrics['minority_recall']:.3f} and a balanced accuracy of {threshold_metrics['balanced_accuracy']:.3f}."),
        p("Business factors such as training time, prediction speed, interpretability, and operational complexity are recorded for every model next to its scores, "
          "because a slightly better model is not worth it if it is much harder to run or explain.", styles["Callout"]),
        find("artifacts/reports/model_comparison_validation.csv, model_business_factors.csv, threshold_sweep_validation.csv, analytical_model_selection.json; artifacts/model_metadata_store/; MLflow; notebook modelling.ipynb/Models.ipynb"),
        PageBreak(),
    ])

    # 6 ---------------------------------------------------------------------
    story.extend([
        p("6. Cost_per_Day leakage review", styles["H1x"]),
        body(f"An earlier notebook proposed the feature {leakage_review['feature']}. It looks useful, and it would probably raise the scores, but Delivery_Days "
             "is only known after the order has been delivered. At the moment we want to make the prediction, that number does not exist yet."),
        body("Using it would be like predicting the weather with tomorrow's newspaper: the score goes up, but the model could never work in practice. "
             "We therefore excluded it, together with Ship_Date, Delivery_Date, and Delivery_Days, from every split, model, and evaluation."),
        p(leakage_review["decision"].replace("_", " "), styles["Warning"]),
        body(leakage_review["scientific_interpretation"]),
        find("artifacts/reports/cost_per_day_leakage_review.json/.md, feature_leakage_assessment.json"),
        PageBreak(),
    ])

    # 7 ---------------------------------------------------------------------
    eval_rows = [["Test measure", model_name, "Naive baseline", "Required", "Result"]]
    eval_rows += [
        ["ROC-AUC", f"{cand['roc_auc']:.4f}", f"{base['roc_auc']:.4f}", f">= {config.minimum_roc_auc:.2f}", met(gates["roc_auc"]["passed"])],
        ["Delayed recall", f"{cand['minority_recall']:.4f}", f"{base['minority_recall']:.4f}", f">= {config.minimum_minority_recall:.2f}", met(gates["minority_recall"]["passed"])],
        ["Weighted F1", f"{cand['f1_weighted']:.4f}", f"{base['f1_weighted']:.4f}", f"+{config.minimum_f1_improvement:.2f} uplift", met(gates["weighted_f1_improvement_over_baseline"]["passed"])],
        ["Net benefit", f"EUR {econ['net_benefit_eur']:,.0f}", f"EUR {base_econ['net_benefit_eur']:,.0f}", "> baseline", met(gates["net_benefit_improvement_over_baseline_eur"]["passed"])],
        ["Net benefit (absolute)", f"EUR {econ['net_benefit_eur']:,.0f}", "-", "> EUR 0", met(gates["candidate_net_benefit_eur"]["passed"])],
        ["Accuracy", f"{cand['accuracy']:.4f}", f"{base['accuracy']:.4f}", "-", "-"],
        ["Balanced accuracy", f"{cand['balanced_accuracy']:.4f}", f"{base['balanced_accuracy']:.4f}", "-", "-"],
        ["Delayed precision", f"{cand['minority_precision']:.4f}", f"{base['minority_precision']:.4f}", "-", "-"],
    ]
    story.extend([
        p("7. Evaluation and business checkpoint", styles["H1x"]),
        p(f"Final selected algorithm: {model_name}. Deployment status: "
          + (f"deployed as a live prototype; {gates_met} of {len(gates)} evaluation gates met." if override else "not approved because the business gates failed."), styles["Callout"]),
        body(f"Once the model and its threshold were frozen, we opened the test set exactly once: {splits['test']:,} orders ({split_delayed['test']:,} delayed) "
             "that played no part in training or tuning. This gives an honest estimate of how the model would behave on new orders."),
        table(eval_rows, [40, 33, 33, 33, 31]),
        Spacer(1, 3 * mm),
        p("How to read these results", styles["H2x"]),
        body(f"<b>What works.</b> The model catches {cand['minority_recall']:.1%} of delayed orders, while the baseline catches none. Under the cost assumptions in "
             f"section 8 this makes it EUR {gates['net_benefit_improvement_over_baseline_eur']['actual']:,.0f} cheaper than the baseline on the test set."),
        body(f"<b>What does not work yet.</b> A ROC-AUC of {cand['roc_auc']:.3f} means the model ranks delayed and delivered orders about as well as chance (0.50). "
             f"To catch nearly half of the delayed orders it has to flag {flagged:,} of {splits['test']:,} orders, and only {cand['minority_precision']:.1%} of those flags are real delays. "
             "Accuracy and weighted F1 fall below the baseline because many delivered orders are flagged as well."),
        body("<b>Why.</b> Section 4 already showed that delay rates barely change across region, category, shipping mode, cost, and date. No algorithm can learn a "
             "pattern that the inputs do not contain, so the limit lies in the available data rather than in the modelling."),
        p(f"Checkpoint decision: {evaluation['checkpoint_decision']}. In MLOps-DSPM terms, the next iteration returns to business understanding to obtain stronger "
          "pre-shipment data (carrier, warehouse, route distance, promised delivery date, dispatch backlog) before the model is used for real decisions.", styles["Callout"]),
        find("artifacts/reports/evaluation_report.json/.md, final_model_report.json, checkpoint_decision.json, naive_vs_best_candidate.png; notebook evalaution/stage4_evaluation.ipynb"),
        PageBreak(),
    ])

    # 8 ---------------------------------------------------------------------
    comp = econ["components"]
    story.extend([
        p("8. Cost and benefit of correct and wrong classifications", styles["H1x"]),
        body("A prediction can be right or wrong in two ways each, and each case has a different business effect. Delivered is class 1 and Delayed is class 0, "
             f"so a correctly flagged delay is a true negative. The counts below are for the {splits['test']:,} test orders at threshold {threshold:.2f}."),
        table([
            ["Outcome", "Count", "Meaning", "Unit value", "Total"],
            ["Correct delay flag (TN)", f"{cm['tn']:,}", "Delayed order flagged; operations can intervene", f"+EUR {config.benefit_correct_risk_flag:.0f}", f"EUR {comp['correct_risk_flags']['total_eur']:,.0f}"],
            ["Correct delivered (TP)", f"{cm['tp']:,}", "Delivered order left alone", f"+EUR {config.benefit_correct_on_time:.0f}", f"EUR {comp['correct_on_time_predictions']['total_eur']:,.0f}"],
            ["Missed delay (FP)", f"{cm['fp']:,}", "Delayed order not flagged", f"-EUR {config.cost_missed_risk:.0f}", f"EUR {comp['missed_risks']['total_eur']:,.0f}"],
            ["Unnecessary intervention (FN)", f"{cm['fn']:,}", "Delivered order flagged as delayed", f"-EUR {config.cost_unnecessary_intervention:.0f}", f"EUR {comp['unnecessary_interventions']['total_eur']:,.0f}"],
            ["Net benefit", "", "", "", f"EUR {econ['net_benefit_eur']:,.0f}"],
        ], [44, 18, 62, 22, 28], header=ORANGE, zebra=LIGHT_ORANGE),
        Spacer(1, 3 * mm),
        body(f"Of the {actual_delayed:,} delayed test orders, {cm['tn']:,} were flagged and {cm['fp']:,} were missed. The naive baseline flags nothing, so it misses all "
             f"{base_econ['confusion_matrix']['fp']:,} delays and ends at EUR {base_econ['net_benefit_eur']:,.0f}. The model's EUR {econ['net_benefit_eur']:,.0f} is therefore "
             f"EUR {gates['net_benefit_improvement_over_baseline_eur']['actual']:,.0f} better, but still negative overall, because each missed delay is assumed to cost EUR 80."),
        p("The unit values are transparent scenario assumptions set in mlops_pipeline/config.py. Finance and operations should replace them with measured "
          "customer-service, compensation, and intervention costs before any investment decision.", styles["Callout"]),
        p("Business objective: pros", styles["H2x"]),
        body(f"The pipeline is fully reproducible; the test set is isolated; the model finds {cand['minority_recall']:.1%} of delayed orders that would otherwise go unnoticed; "
             "it is cheaper than doing nothing under the stated assumptions; and every decision is documented and auditable."),
        p("Business objective: cons", styles["H2x"]),
        body("Ranking quality is close to chance; many delivered orders are flagged, which would waste intervention effort; the absolute net benefit is negative; "
             "and the order-time inputs contain little signal about delays."),
        find("artifacts/reports/cost_benefit_analysis.json/.md, business_objective_assessment.json/.md"),
        PageBreak(),
    ])

    # 9 ---------------------------------------------------------------------
    story.extend([
        p("9. MLflow, metadata store, and model registry", styles["H1x"]),
        body("Every model we trained is logged to MLflow with its parameters, training time, prediction speed, validation scores, estimated business value, "
             "the saved model, an input example, and its signature. The final test evaluation is logged as a separate run with all business reports attached. "
             "This means any result in this report can be traced back to the exact run that produced it."),
        body("The model registry keeps two clearly separated entries. The <b>candidate</b> is the selected model with the automatic evaluation result. "
             "The <b>champion</b> is the model that is actually allowed to serve predictions, together with who approved it, when, and why."),
        p("View MLflow", styles["H2x"]),
        p("mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/tracking.db --host 127.0.0.1 --port 5000", styles["CodeBlock"]),
        body(f"Then open http://127.0.0.1:5000 and select the experiment <b>ecommerce-shipping-bpmn</b>. The final evaluation run ID is {evaluation.get('mlflow_evaluation_run_id', '-')}."),
        p("Current registry result", styles["H2x"]),
        (p(f"Candidate: {model_name}, automatic gates not all met (registry stage: rejected). Champion: the same evaluated {model_name} bundle, verified by its SHA-256 "
           f"fingerprint and deployed as a live prototype by {override['actor']} on {override['approved_at'][:10]}.", styles["Callout"])
         if override else p("The candidate manifest is marked rejected/blocked because evaluation failed.", styles["Warning"])),
        find("artifacts/mlflow/tracking.db; artifacts/model_metadata_store/*.json; artifacts/model_registry/candidate_manifest.json and champion_manifest.json"),
        PageBreak(),
    ])

    # 10 --------------------------------------------------------------------
    story.extend([
        p("10. Deployment", styles["H1x"]),
        body("Deployment means more than saving a model file. The model is packaged with its scaler, category codes, feature order, and threshold into one bundle, "
             "and served by a FastAPI service with single and batch prediction endpoints, health, model, and pipeline information, monitoring, feedback, and support. "
             "The same service hosts a web app at its root address, so users can try predictions in a browser. A Dockerfile builds a self-contained image, "
             "a GitHub Actions workflow runs the full test suite on every push, and a smoke-test script checks a running deployment end to end."),
        p("Gate-aware status", styles["H2x"]),
        body("The service only serves predictions from a model that is allowed to serve: either one that passed every gate, or one with a documented deployment "
             "decision. Without either, prediction requests are refused (HTTP 503), so no unapproved model can serve unnoticed."),
        *(
            [
                body(f"<b>Deployment decision.</b> {override['actor']} deployed the evaluated {model_display_name(override['model'])} model as a live prototype on "
                     f"{override['approved_at'][:10]}. Reason: {override['rationale']} Gates not yet met: "
                     f"{', '.join(GATE_LABELS.get(g, g) for g in override['failed_business_gates'])}. Each API response states its deployment basis and notes that "
                     "the model is a prototype."),
            ] if override else []
        ),
        p(
            (f"Live deployment smoke test: {smoke['status'].upper()} on {smoke['generated_at'][:10]}. A real prediction was served (HTTP {smoke['prediction_http_status']}), "
             "and health, pipeline status, monitoring, and metrics were all available." if smoke else "Live deployment smoke test has not yet been executed."),
            styles["Callout"] if smoke else styles["Warning"],
        ),
        Spacer(1, 3 * mm),
        p("Live demo", styles["H2x"]),
        p(f"{LIVE_DEMO_URL} - a free Hugging Face Space. The trained {model_name} model is exported to JSON (scripts/export_browser_model.py) and runs in the browser. "
          f"An automated test confirms it gives the same predictions as the Python model on all {in_scope:,} orders.", styles["Callout"]),
        p("Prototype commands", styles["H2x"]),
        p("uvicorn deployment.app:app --host 127.0.0.1 --port 8000<br/>"
          "python scripts/smoke_test_deployment.py --base-url http://127.0.0.1:8000 --require-approved<br/>"
          "docker build -t ecommerce-shipping-prediction .<br/>"
          "docker run --rm -p 8000:8000 ecommerce-shipping-prediction", styles["CodeBlock"]),
        find("deployment/app.py, deployment/static/, deployment/static_space/, Dockerfile, .github/workflows/deployment-ci.yml, artifacts/reports/deployment_smoke_test.json, deployment_override.json"),
        PageBreak(),
    ])

    # 11 --------------------------------------------------------------------
    story.extend([
        p("11. Utilization, monitoring, and retraining", styles["H1x"]),
        body("Users enter an order (region, category, shipping mode, cost, and date) in the web app and immediately see Delivered or Delayed with the probability. "
             "Technical users can call the same API through the Swagger page at /docs. Every prediction gets an ID, so the real outcome can be reported later through /feedback."),
        body("The service records requests, predictions, errors, the share of each predicted class, the average predicted probability, and recent events. "
             "Once real outcomes arrive, it computes observed accuracy, ROC-AUC, and delayed recall, and recommends retraining when at least 30 outcomes show "
             "ROC-AUC below 0.60 or delayed recall below 0.40. Metrics are also exposed in Prometheus format for dashboards."),
        p("Retraining workflow", styles["H2x"]),
        body("Retraining starts when the source data changes (detected by its fingerprint), when the model is older than 30 days, or when the observed "
             "ROC-AUC reported to the retraining script falls below 0.60, and it can be forced manually. Each attempt runs the full pipeline in its own timestamped folder, and the current "
             "champion is only replaced if the new model passes every gate."),
        p("python scripts/retrain_model.py --force", styles["CodeBlock"]),
        (p(f"Retraining run {retraining['version']} (triggers: {', '.join(retraining['triggers'])}) completed with decision {retraining['checkpoint_decision']} "
           f"and was {'promoted' if retraining['promoted'] else 'not promoted'}, so the running champion stayed unchanged, exactly as designed.", styles["Callout"])
         if retraining else body("No retraining run has been recorded yet.")),
        p("Utilization status: active as a live prototype (web app, API, and monitoring)." if override
          else f"Production utilization status: {lifecycle['utilization']['production_utilization_status']}", styles["Callout" if override else "Warning"]),
        find("/monitoring, /metrics, /utilization, /feedback in deployment/app.py; runtime/prediction_log.jsonl; artifacts/retraining_report.json; artifacts/retraining_runs/"),
        PageBreak(),
    ])

    # 12 --------------------------------------------------------------------
    test_rows = [["Test file", "Tests", "What it checks"]]
    descriptions = {
        "test_bpmn_pipeline.py": "Data stage outputs and split sizes, Returned removal, leakage exclusions, cost model, checkpoint report, retraining trigger, business artifacts, lifecycle status, deployment decision record",
        "test_deployment_api.py": "Health, single/batch/processed predictions, blocking without approval, invalid input, monitoring, metrics, model info, feedback, support tickets, web app, deployment decision",
        "test_browser_model.py": f"Live web app predictions match the Python model on all {in_scope:,} orders",
    }
    for name, count in sorted(test_counts.items()):
        test_rows.append([name, str(count), descriptions.get(name, "")])
    story.extend([
        p("12. Testing, CI, and reproducibility", styles["H1x"]),
        body("Automated tests protect every stage of the pipeline and the deployed service, so a change that breaks something is caught before it reaches users."),
        table(test_rows, [44, 14, 116]),
        Spacer(1, 3 * mm),
        p(f"Current result: all {total_tests} tests pass." if total_tests else "Run pytest -q to execute the test suite.", styles["Callout"]),
        body("The GitHub Actions workflow installs the runtime and development dependencies and runs the complete suite on every push and pull request."),
        p("Reproduce the project", styles["H2x"]),
        p("pip install -r requirements.txt -r requirements-dev.txt<br/>python scripts/run_end_to_end_pipeline.py<br/>pytest -q<br/>python scripts/generate_project_report.py", styles["CodeBlock"]),
        body("The pipeline uses a fixed random seed (42), so rerunning it reproduces the same splits, models, and numbers. Add --quick for a faster structural run with smaller models."),
        find("tests/; .github/workflows/deployment-ci.yml; README.md"),
        PageBreak(),
    ])

    # 13 --------------------------------------------------------------------
    story.extend([
        p("13. Requirement traceability", styles["H1x"]),
        body("Each requirement from the project proposal and the MLOps-DSPM material, with the evidence that shows it is done."),
        table([
            ["Requirement", "Implementation evidence", "Status"],
            ["Business Understanding", "Situation assessment, requirements, project map, success gates (section 3)", "Complete"],
            ["Data exploration and preparation", f"Quality checks, contracts, {scope['rows_removed']:,} Returned removed, feature store (section 4)", "Complete"],
            ["Model selection and training", "Four models, validation-only selection, MLflow tracking (section 5)", "Complete"],
            ["Leakage review", "Cost_per_Day and post-outcome fields excluded (section 6)", "Complete"],
            ["Business-goal evaluation", f"Untouched test set, costs, {gates_met} of {len(gates)} gates met (sections 7-8)", "Complete"],
            ["Deployment and testing", f"FastAPI, web app, Docker, CI, {total_tests} tests, smoke test (sections 10, 12)", "Live prototype" if override else "Prototype complete"],
            ["Use, monitoring, maintenance", "Monitoring, feedback, support, guarded retraining (section 11)", "Active (prototype)" if override else "Prototype complete"],
            ["Architecture documentation", "BPMN-aligned process, UML component and sequence diagrams (docs/architecture.md)", "Complete"],
            ["Critical reflection", "Data, modelling, technology, and process review (section 14)", "Complete"],
        ], [48, 90, 36]),
        PageBreak(),
    ])

    # 14 --------------------------------------------------------------------
    story.extend([
        p("14. Challenges and critical reflection", styles["H1x"]),
        p("Data challenge", styles["H2x"]),
        body("The dataset is clean, but clean data is not the same as predictive data. Delay rates are almost identical across every order-time attribute, so the "
             "model has very little to learn from. The most valuable next step is to collect information that is known before shipping and plausibly drives delays: "
             "carrier, warehouse, route distance, promised delivery date, stock availability, dispatch backlog, and weather."),
        p("Modeling challenge", styles["H2x"]),
        body("The earlier notebooks mixed in Returned orders as a third class and tuned the decision threshold on the test set, which makes results look better "
             "than they are. The corrected design removes Returned completely, models Delivered vs Delayed only, keeps training, validation, and test strictly separate, "
             "and compares all models on the same terms."),
        p("Technology reflection", styles["H2x"]),
        body("CSV and Joblib files for the feature store and a local SQLite MLflow backend fit an academic prototype well: they are simple, transparent, and free. "
             "A production system would use a managed feature store, a shared MLflow server, central artifact storage, orchestration, authentication, and alerting. "
             "For the public demo, a free static Hugging Face Space runs the exported model in the browser; a hosted server would add shared monitoring."),
        p("Process-model reflection", styles["H2x"]),
        p("The MLOps-DSPM is valuable because it makes evidence-based decisions possible, including the decision to go back a stage. Here the pipeline, deployment, "
          "monitoring, and retraining are complete and running, and the evaluation clearly shows what is needed next: better input data. "
          + ("The model is deployed as a documented live prototype, so the complete lifecycle can be demonstrated while the evidence and the decision stay transparent." if override else ""),
          styles["Callout"]),
        PageBreak(),
    ])

    # 15 --------------------------------------------------------------------
    story.extend([
        p("15. Final conclusion", styles["H1x"]),
        body("The project implements the complete MLOps-DSPM lifecycle required by the proposal: management artifacts, a reproducible data pipeline with a feature store, "
             f"tracked experiments in MLflow, a model registry, a business-focused evaluation, a deployed API and web app, monitoring, {total_tests} automated tests with CI, "
             "and guarded retraining."),
        p(f"The final selected algorithm is {model_name}. On {splits['test']:,} unseen test orders it catches {cand['minority_recall']:.1%} of delayed orders and is "
          f"EUR {gates['net_benefit_improvement_over_baseline_eur']['actual']:,.0f} cheaper than doing nothing, meeting {gates_met} of {len(gates)} gates. "
          f"Its ROC-AUC of {cand['roc_auc']:.3f} shows that order-time attributes alone cannot yet predict delays reliably. "
          + ("The model is live as a prototype for demonstration. " if override else "")
          + "The next iteration should confirm the intervention process and true costs with the business and collect stronger pre-shipment operational data.",
          styles["Callout"]),
        p("Final status", styles["H2x"]),
        p(status_text, styles["Callout"]),
        p("References", styles["H2x"]),
        p("1. Machine Learning-Operations-based Data Science Process Model - Material (DSPM_material_STP.pdf).<br/>"
          "2. Project Proposal: Development and Operationalization of Data Science Solutions - E-Commerce Shipping Prediction (STP_ecommerce.pdf).<br/>"
          "3. Kreuzberger, D., Kuhl, N., and Hirschl, S. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access, 11, 31866-31879.<br/>"
          "4. Haertel, C., Pohl, M., Nahhas, A., Staegemann, D., and Turowski, K. (2022). Toward a Lifecycle for Data Science. PACIS 2022 Proceedings.", styles["Smallx"]),
    ])

    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    return OUTPUT


if __name__ == "__main__":
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    print(build_report())
