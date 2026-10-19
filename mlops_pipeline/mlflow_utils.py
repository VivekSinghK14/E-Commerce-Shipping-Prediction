from pathlib import Path

import mlflow

from .config import PipelineConfig


EXPERIMENT_NAME = "ecommerce-shipping-bpmn"


def configure_mlflow(config: PipelineConfig) -> str:
    """Configure a durable SQLite backend and local artifact store."""
    config.mlflow_store.mkdir(parents=True, exist_ok=True)
    database = (config.mlflow_store / "tracking.db").resolve()
    tracking_uri = f"sqlite:///{database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    artifact_root = (config.mlflow_store / "artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_root.as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri
