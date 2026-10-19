import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops_pipeline.config import PipelineConfig
from mlops_pipeline.retraining import run_retraining


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guarded, versioned model retraining.")
    parser.add_argument("--force", action="store_true", help="Retrain even when automatic triggers are absent.")
    parser.add_argument("--quick", action="store_true", help="Use smaller ensembles for a smoke run.")
    parser.add_argument("--observed-roc-auc", type=float, default=None)
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--raw-data", type=Path, default=None)
    args = parser.parse_args()
    defaults = PipelineConfig()
    config = PipelineConfig(
        raw_data=(args.raw_data or defaults.raw_data).resolve(),
        artifacts=(args.artifacts_dir or defaults.artifacts).resolve(),
    )
    result = run_retraining(
        config,
        force=args.force,
        quick=args.quick,
        observed_roc_auc=args.observed_roc_auc,
        max_age_days=args.max_age_days,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
