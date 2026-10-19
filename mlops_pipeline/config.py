from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineConfig:
    """All paths and evaluation gates used by the pipeline."""

    raw_data: Path = ROOT / "50K Records.csv"
    artifacts: Path = ROOT / "artifacts"
    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20
    # The candidate must show useful discrimination and beat the honest baseline.
    minimum_roc_auc: float = 0.60
    minimum_minority_recall: float = 0.40
    minimum_f1_improvement: float = 0.02
    minimum_net_benefit_improvement: float = 0.0
    minimum_candidate_net_benefit: float = 0.0
    # Illustrative EUR values. Replace them with finance/operations estimates.
    benefit_correct_risk_flag: float = 30.0
    benefit_correct_on_time: float = 2.0
    cost_missed_risk: float = 80.0
    cost_unnecessary_intervention: float = 10.0
    mlflow_tracking_dir: Path | None = None

    @property
    def feature_store(self) -> Path:
        return self.artifacts / "feature_store"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    @property
    def model_registry(self) -> Path:
        return self.artifacts / "model_registry"

    @property
    def metadata_store(self) -> Path:
        return self.artifacts / "model_metadata_store"

    @property
    def mlflow_store(self) -> Path:
        return self.mlflow_tracking_dir or (self.artifacts / "mlflow")

    @property
    def retraining_runs(self) -> Path:
        return self.artifacts / "retraining_runs"


FEATURE_COLUMNS = [
    "Customer_Region",
    "Product_Category",
    "Shipping_Mode",
    "Shipping_Cost",
    "Order_Month",
    "Order_DayOfWeek",
]

NUMERIC_COLUMNS = [
    "Shipping_Cost",
    "Order_Month",
    "Order_DayOfWeek",
]

REQUIRED_RAW_COLUMNS = [
    "Order_ID",
    "Customer_Region",
    "Product_Category",
    "Order_Date",
    "Ship_Date",
    "Delivery_Date",
    "Shipping_Mode",
    "Shipping_Cost",
    "Delivery_Status",
    "Delivery_Days",
]

# Project scope: binary Delivered vs Delayed. Returned orders are a different
# business process (reverse logistics), so they are removed at ingestion and
# never reach features, targets, models, reports, or the serving API.
TARGET_COLUMN = "Delivery_Status"
POSITIVE_OUTCOME = "Delivered"
NEGATIVE_OUTCOME = "Delayed"
TARGET_OUTCOMES = [POSITIVE_OUTCOME, NEGATIVE_OUTCOME]
TARGET_MAP = {POSITIVE_OUTCOME: 1, NEGATIVE_OUTCOME: 0}
OUT_OF_SCOPE_OUTCOMES = ["Returned"]
