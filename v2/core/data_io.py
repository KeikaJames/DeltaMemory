"""Bank data I/O — load Exp35b/38 bank.pt and prep b-vectors for preload."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import torch

REPO = Path(__file__).resolve().parents[2]
BANK_PT_DEFAULT = REPO / "v1/experiments/atb_validation_v1/exp35b_memit_bank/data/bank.pt"


def load_bank_blob(path: str | Path = BANK_PT_DEFAULT):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def filter_keys(entries: dict, *, split: str | None = None, solo_pass: bool | None = None) -> list:
    keys = list(entries.keys())
    if split is not None:
        keys = [k for k in keys if entries[k].get("split") == split]
    if solo_pass is not None:
        keys = [k for k in keys if entries[k].get("solo_pass") == solo_pass]
    return keys


def items_for_keys(entries: dict, keys: Iterable[str]) -> list[tuple[str, str, str]]:
    return [(entries[k]["subject"], entries[k]["relation"], entries[k]["target_true"]) for k in keys]


def b_stack_for_keys(entries: dict, keys: Iterable[str], *, target_norm: float | None = 15.0,
                     device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Stack and optionally L2-renormalize the residual b-vectors."""
    b = torch.stack([entries[k]["b"].float() for k in keys], dim=0)
    if target_norm is not None:
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-9) * target_norm
    return b.to(device=device, dtype=dtype)


def relation_of(entries: dict, key: str) -> str:
    return entries[key].get("relation", "")


def split_disjoint_relations(entries: dict, *, train_frac: float = 0.7, seed: int = 0) -> tuple[list, list]:
    """Return (train_relations, test_relations) lists with empty intersection."""
    rels = sorted({entries[k].get("relation", "") for k in entries})
    rng = random.Random(seed)
    rng.shuffle(rels)
    n_train = int(len(rels) * train_frac)
    return rels[:n_train], rels[n_train:]
