"""Compatibility entry point for the authoritative end-to-end pipeline.

The earlier version retrained a leakage-prone legacy XGBoost artifact from the
``processed`` directory. Production runs now use the same validated, gated
lifecycle as every other entry point.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_end_to_end_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
