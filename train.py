"""
IndianRoadDetector (IRD) Training Entrypoint.

Delegates to scripts/train_custom.py to provide a root-level training script.
Supports both IRD V1.5 and IRD V2.

NOTE: Do NOT run full training on low-resource local hardware.
This entrypoint is provided for execution on cloud GPU / cluster environments.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.train_custom import main

if __name__ == "__main__":
    main()
