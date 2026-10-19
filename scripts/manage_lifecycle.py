import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops_pipeline.config import PipelineConfig
from mlops_pipeline.lifecycle_stage import manage_system_lifecycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a DSPM utilization lifecycle decision.")
    parser.add_argument("action", choices=["status", "discontinue", "resume"])
    parser.add_argument("--actor", required=True, help="Accountable role or decision-maker.")
    parser.add_argument("--reason", required=True, help="Decision rationale for the audit log.")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()
    defaults = PipelineConfig()
    config = PipelineConfig(artifacts=(args.artifacts_dir or defaults.artifacts).resolve())
    result = manage_system_lifecycle(config, args.action, args.actor, args.reason)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())