import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops_pipeline.config import PipelineConfig
from mlops_pipeline.data_stage import collect_and_prepare
from mlops_pipeline.evaluation_stage import evaluate_and_checkpoint
from mlops_pipeline.model_stage import develop_and_assess_models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BPMN stages 3-5 from raw CSV through the evaluation checkpoint."
    )
    parser.add_argument("--raw-data", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Use smaller ensembles for a smoke run.")
    args = parser.parse_args()

    defaults = PipelineConfig()
    config = PipelineConfig(
        raw_data=(args.raw_data or defaults.raw_data).resolve(),
        artifacts=(args.artifacts_dir or defaults.artifacts).resolve(),
    )

    print("[1/3] Data collection, exploration, and preparation")
    collect_and_prepare(config)
    print("[2/3] Analysis/modeling")
    develop_and_assess_models(config, quick=args.quick)
    print("[3/3] Evaluation and checkpoint")
    evaluation = evaluate_and_checkpoint(config)
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
