import pandas as pd

from .config import FEATURE_COLUMNS


CATEGORICAL_COLUMNS = ["Customer_Region", "Product_Category", "Shipping_Mode"]


def build_order_time_features(
    records: pd.DataFrame, category_maps: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """Build only fields available when the order is placed."""
    features = pd.DataFrame(index=records.index)
    for column in CATEGORICAL_COLUMNS:
        normalized_map = {
            str(label).strip().casefold(): code
            for label, code in category_maps[column].items()
        }
        normalized_values = records[column].astype(str).str.strip().str.casefold()
        unknown = sorted(set(normalized_values) - set(normalized_map))
        if unknown:
            raise ValueError(f"Unknown {column} values: {unknown}")
        features[column] = normalized_values.map(normalized_map)

    features["Shipping_Cost"] = pd.to_numeric(records["Shipping_Cost"], errors="raise")
    has_calendar_fields = {"Order_Month", "Order_DayOfWeek"}.issubset(records.columns)
    if has_calendar_fields and records[["Order_Month", "Order_DayOfWeek"]].notna().all(axis=None):
        features["Order_Month"] = records["Order_Month"].astype(int)
        features["Order_DayOfWeek"] = records["Order_DayOfWeek"].astype(int)
    else:
        if "Order_Date" not in records:
            raise ValueError("Provide Order_Date or both Order_Month and Order_DayOfWeek")
        order_dates = pd.to_datetime(records["Order_Date"], format="mixed", errors="coerce")
        if order_dates.isna().any():
            raise ValueError("Order_Date contains invalid values")
        features["Order_Month"] = order_dates.dt.month
        features["Order_DayOfWeek"] = order_dates.dt.dayofweek
    return features[FEATURE_COLUMNS]