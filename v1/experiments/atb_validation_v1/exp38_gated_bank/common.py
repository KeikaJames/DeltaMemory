"""Exp38 — common loaders shared by all variant scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: F401,E402
from gates import BankTensors, stack_bank  # noqa: E402


def load_bank(path: Path | str, device: str, dtype) -> BankTensors:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    # support either {"entries": {...}} or direct {id: {a,b}}
    if isinstance(raw, dict) and "entries" in raw:
        entries = raw["entries"]
    else:
        entries = raw
    return stack_bank(entries, device=device, dtype=dtype)


def get_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
