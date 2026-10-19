import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops_pipeline.business_understanding import document_business_understanding
from mlops_pipeline.config import PipelineConfig
from mlops_pipeline.data_stage import collect_and_prepare
from mlops_pipeline.evaluation_stage import evaluate_and_checkpoint
from mlops_pipeline.lifecycle_stage import assess_deployment_and_utilization
from mlops_pipeline.model_stage import develop_and_assess_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete MLOps-DSPM lifecycle.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--raw-data", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()
    defaults = PipelineConfig()
    config = PipelineConfig(
        raw_data=(args.raw_data or defaults.raw_data).resolve(),
        artifacts=(args.artifacts_dir or defaults.artifacts).resolve(),
    )

    print("[1/6] Business Understanding")
    document_business_understanding(config)
    print("[2/6] Data Collection, Exploration, and Preparation")
    collect_and_prepare(config)
    print("[3/6] Analysis and Modeling")
    develop_and_assess_models(config, quick=args.quick)
    print("[4/6] Evaluation")
    evaluation = evaluate_and_checkpoint(config)
    print("[5/6] Deployment readiness")
    lifecycle = assess_deployment_and_utilization(config, evaluation)
    print("[6/6] Utilization and maintenance readiness")
    print(json.dumps(lifecycle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
