"""Standardized model loader for v2.

Re-exports ``atb_validation_v1._lib.load_model`` so v2 experiments don't need
sys.path gymnastics. Kept as a thin shim so we can later swap the loader
implementation without touching call sites.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))

from atb_validation_v1._lib import load_model  # noqa: E402

__all__ = ["load_model"]
