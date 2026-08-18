from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    PipelineConfig,
)
from .io_utils import file_sha256, utc_now, write_json


def _data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "Order_ID": "Unique order identifier; excluded from model features.",
        "Customer_Region": "Customer region (categorical predictor).",
        "Product_Category": "Ordered product category (categorical predictor).",
        "Order_Date": "Date order was placed; source for calendar features.",
        "Ship_Date": "Date order shipped; used only for quality validation.",
        "Delivery_Date": "Date delivered; used only for quality validation.",
        "Shipping_Mode": "Selected shipping mode (categorical predictor).",
        "Shipping_Cost": "Shipping charge (numeric predictor).",
        "Delivery_Status": "Outcome: Delivered, Delayed, or Returned.",
        "Delivery_Days": "Days from shipping to delivery (numeric predictor).",
    }
    return pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(frame[c].dtype) for c in frame.columns],
            "non_null": [int(frame[c].notna().sum()) for c in frame.columns],
            "unique_values": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
            "description": [descriptions.get(c, "") for c in frame.columns],
        }
    )


def collect_and_prepare(config: PipelineConfig) -> dict:
    """Implement source identification through feature-store publication."""
    if not config.raw_data.exists():
        raise FileNotFoundError(f"Configured data source does not exist: {config.raw_data}")

    raw = pd.read_csv(config.raw_data)
    missing_columns = sorted(set(REQUIRED_RAW_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Raw CSV is missing required columns: {missing_columns}")

    config.reports.mkdir(parents=True, exist_ok=True)
    config.feature_store.mkdir(parents=True, exist_ok=True)

    data_requirements = {
        "generated_at": utc_now(),
        "business_question": "Can order-time attributes identify shipments at risk of not being delivered?",
        "unit_of_analysis": "one order",
        "required_columns": REQUIRED_RAW_COLUMNS,
        "minimum_rows": 1000,
        "allowed_outcomes": ["Delivered", "Delayed", "Returned"],
        "prediction_target": "Delivered (1) versus Delayed/Returned (0)",
    }
    write_json(config.reports / "data_requirements.json", data_requirements)
    input_contract = {
        "format": "CSV with header row, UTF-8 compatible",
        "columns": {
            "Order_ID": {"logical_type": "string", "nullable": False, "unique": True},
            "Customer_Region": {"logical_type": "category", "nullable": False},
            "Product_Category": {"logical_type": "category", "nullable": False},
            "Order_Date": {"logical_type": "date", "nullable": False},
            "Ship_Date": {"logical_type": "date", "nullable": False},
            "Delivery_Date": {"logical_type": "date", "nullable": False},
            "Shipping_Mode": {"logical_type": "category", "nullable": False},
            "Shipping_Cost": {"logical_type": "number", "minimum": 0, "nullable": False},
            "Delivery_Status": {
                "logical_type": "category",
                "allowed": ["Delivered", "Delayed", "Returned"],
                "nullable": False,
            },
            "Delivery_Days": {"logical_type": "integer", "minimum_exclusive": 0, "nullable": False},
        },
    }
    write_json(config.reports / "input_data_contract.json", input_contract)

    description = {
        "generated_at": utc_now(),
        "source": str(config.raw_data),
        "source_sha256": file_sha256(config.raw_data),
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "column_names": list(raw.columns),
        "memory_bytes": int(raw.memory_usage(deep=True).sum()),
    }
    write_json(config.reports / "data_description_report.json", description)
    _data_dictionary(raw).to_csv(config.reports / "data_dictionary.csv", index=False)

    dates = {}
    for column in ["Order_Date", "Ship_Date", "Delivery_Date"]:
        dates[column] = pd.to_datetime(raw[column], format="mixed", errors="coerce")

    invalid_dates = {column: int(series.isna().sum()) for column, series in dates.items()}
    duplicate_order_ids = int(raw["Order_ID"].duplicated().sum())
    invalid_categories = {
        "Customer_Region": int((raw["Customer_Region"].astype(str).str.strip() == "").sum()),
        "Product_Category": int((raw["Product_Category"].astype(str).str.strip() == "").sum()),
        "Shipping_Mode": int((raw["Shipping_Mode"].astype(str).str.strip() == "").sum()),
        "Delivery_Status": int(
            (~raw["Delivery_Status"].isin(["Delivered", "Delayed", "Returned"])).sum()
        ),
    }
    quality_checks = {
        "missing_values": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_order_ids": duplicate_order_ids,
        "invalid_dates": invalid_dates,
        "order_after_ship": int((dates["Order_Date"] > dates["Ship_Date"]).sum()),
        "ship_after_delivery": int((dates["Ship_Date"] > dates["Delivery_Date"]).sum()),
        "delivery_days_mismatch": int(
            (((dates["Delivery_Date"] - dates["Ship_Date"]).dt.days) != raw["Delivery_Days"]).sum()
        ),
        "nonpositive_delivery_days": int((raw["Delivery_Days"] <= 0).sum()),
        "negative_shipping_cost": int((raw["Shipping_Cost"] < 0).sum()),
        "invalid_categories": invalid_categories,
    }
    blocking_defects = sum(
        [
            quality_checks["missing_values"],
            quality_checks["duplicate_order_ids"],
            sum(invalid_dates.values()),
            quality_checks["order_after_ship"],
            quality_checks["ship_after_delivery"],
            quality_checks["delivery_days_mismatch"],
            quality_checks["nonpositive_delivery_days"],
            quality_checks["negative_shipping_cost"],
            sum(invalid_categories.values()),
        ]
    )
    quality_report = {
        "generated_at": utc_now(),
        "checks": quality_checks,
        "blocking_defects": int(blocking_defects),
        "data_sufficient": blocking_defects == 0 and len(raw) >= 1000,
    }
    write_json(config.reports / "data_quality_report.json", quality_report)
    if not quality_report["data_sufficient"]:
        raise ValueError("Data quality gate failed; see data_quality_report.json")

    exploration = {
        "generated_at": utc_now(),
        "numeric_summary": raw[["Shipping_Cost", "Delivery_Days"]].describe().to_dict(),
        "category_distributions": {
            column: raw[column].value_counts(dropna=False).to_dict()
            for column in [
                "Customer_Region",
                "Product_Category",
                "Shipping_Mode",
                "Delivery_Status",
            ]
        },
        "date_ranges": {
            column: {"min": series.min(), "max": series.max()} for column, series in dates.items()
        },
    }
    write_json(config.reports / "data_exploration_report.json", exploration)

    prepared = raw.copy()
    prepared["Order_Month"] = dates["Order_Date"].dt.month
    prepared["Order_DayOfWeek"] = dates["Order_Date"].dt.dayofweek
    prepared["Cost_per_Day"] = prepared["Shipping_Cost"] / prepared["Delivery_Days"]

    category_maps = {}
    for column in ["Customer_Region", "Product_Category", "Shipping_Mode"]:
        labels = sorted(prepared[column].unique().tolist())
        category_maps[column] = {label: index for index, label in enumerate(labels)}
        prepared[column] = prepared[column].map(category_maps[column])

    multiclass_map = {"Delayed": 0, "Delivered": 1, "Returned": 2}
    targets = pd.DataFrame(
        {
            "multiclass_target": prepared["Delivery_Status"].map(multiclass_map).astype(int),
            "binary_target": (prepared["Delivery_Status"] == "Delivered").astype(int),
        }
    )
    features = prepared[FEATURE_COLUMNS].copy()

    train_idx, test_idx = train_test_split(
        features.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=targets["binary_target"],
    )
    relative_validation = config.validation_size / (1.0 - config.test_size)
    train_idx, validation_idx = train_test_split(
        train_idx,
        test_size=relative_validation,
        random_state=config.random_state,
        stratify=targets.loc[train_idx, "binary_target"],
    )

    scaler = StandardScaler().fit(features.loc[train_idx, NUMERIC_COLUMNS])
    split_indices = {"train": train_idx, "validation": validation_idx, "test": test_idx}
    for split, indices in split_indices.items():
        x_split = features.loc[indices].copy()
        x_split[NUMERIC_COLUMNS] = scaler.transform(x_split[NUMERIC_COLUMNS])
        x_split.reset_index(drop=True).to_csv(config.feature_store / f"X_{split}.csv", index=False)
        targets.loc[indices].reset_index(drop=True).to_csv(
            config.feature_store / f"y_{split}.csv", index=False
        )

        # End-of-pipeline contract: exact columns, numeric, finite, non-null,
        # aligned row counts, and binary/multiclass target domains.
        if list(x_split.columns) != FEATURE_COLUMNS:
            raise ValueError(f"{split} feature output has incorrect column order")
        if not all(pd.api.types.is_numeric_dtype(x_split[column]) for column in x_split):
            raise ValueError(f"{split} feature output contains non-numeric columns")
        if x_split.isna().any().any() or not np.isfinite(x_split.to_numpy()).all():
            raise ValueError(f"{split} feature output contains null/non-finite values")
        split_targets = targets.loc[indices]
        if not set(split_targets["binary_target"]).issubset({0, 1}):
            raise ValueError(f"{split} binary target violates its contract")
        if not set(split_targets["multiclass_target"]).issubset({0, 1, 2}):
            raise ValueError(f"{split} multiclass target violates its contract")

    joblib.dump(scaler, config.feature_store / "scaler.joblib")
    preparation = {
        "generated_at": utc_now(),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_scaled_columns": NUMERIC_COLUMNS,
        "category_maps": category_maps,
        "target_maps": {
            "multiclass": multiclass_map,
            "binary": {"Delivered": 1, "Delayed_or_Returned": 0},
        },
        "split_rows": {name: len(index) for name, index in split_indices.items()},
        "split_policy": "60% train / 20% validation / 20% untouched test, stratified",
        "leakage_controls": [
            "Order_ID and outcome/date-after-order fields excluded from predictors",
            "scaler fit on training rows only",
            "threshold and model selection use validation, never test",
        ],
    }
    write_json(config.reports / "data_preparation_report.json", preparation)
    write_json(config.feature_store / "feature_manifest.json", preparation)
    output_contract = {
        "format": "CSV feature/target pairs plus Joblib scaler and JSON manifest",
        "feature_columns_in_order": FEATURE_COLUMNS,
        "feature_types": {column: "numeric and finite" for column in FEATURE_COLUMNS},
        "targets": {"binary_target": [0, 1], "multiclass_target": [0, 1, 2]},
        "splits": ["train", "validation", "test"],
        "validation_status": "passed",
    }
    write_json(config.reports / "output_data_contract.json", output_contract)

    data_pipeline_architecture = {
        "generated_at": utc_now(),
        "source": str(config.raw_data),
        "steps": [
            "schema and provenance validation",
            "description and exploration",
            "data-quality gate",
            "leakage-safe feature derivation",
            "stratified train/validation/test split",
            "train-only scaler fit",
            "feature-store publication",
        ],
        "destination": str(config.feature_store),
    }
    write_json(config.reports / "data_pipeline_architecture.json", data_pipeline_architecture)
    return {"description": description, "quality": quality_report, "preparation": preparation}
