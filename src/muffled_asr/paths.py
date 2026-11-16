"""Project root used to resolve data, model and result paths (override with the MSR_ROOT environment variable)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("MSR_ROOT", REPO_ROOT)).expanduser().resolve()
