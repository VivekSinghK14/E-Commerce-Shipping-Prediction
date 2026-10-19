"""Deploy the evaluated candidate through a recorded management override.

The evaluation checkpoint is left unchanged. The override is an accountable,
audited decision that is exposed by every API response.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops_pipeline.config import PipelineConfig
from mlops_pipeline.lifecycle_stage import approve_deployment_override


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the evaluated candidate by override.")
    parser.add_argument("--actor", required=True, help="Accountable decision-maker.")
    parser.add_argument("--reason", required=True, help="Why deployment proceeds despite failed gates.")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()
    defaults = PipelineConfig()
    config = PipelineConfig(artifacts=(args.artifacts_dir or defaults.artifacts).resolve())
    champion = approve_deployment_override(config, args.actor, args.reason)
    print(json.dumps(champion["override"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
