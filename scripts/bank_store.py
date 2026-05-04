#!/usr/bin/env python
"""Phase R-6 — small CLI for inspecting / managing persisted banks.

Usage:
  python scripts/bank_store.py list  --root bank/
  python scripts/bank_store.py info  --root bank/ --model google/gemma-4-E2B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deltamemory.memory.bank_persistence import (  # noqa: E402
    list_banks,
    storage_bytes,
)


def cmd_list(args):
    locs = list_banks(args.root, model_name=args.model)
    if not locs:
        print(f"[bank_store] no banks under {args.root}")
        return
    print(f"{'model':40s}  {'config_sha':18s}  {'n_facts':>8s}  {'bytes':>12s}")
    print("-" * 84)
    for loc in locs:
        meta = json.loads(loc.meta_path.read_text())
        print(f"{meta['model_name']:40s}  {loc.config_sha:18s}  "
              f"{meta['n_facts']:>8d}  {storage_bytes(loc):>12d}")


def cmd_info(args):
    locs = list_banks(args.root, model_name=args.model)
    for loc in locs:
        meta = json.loads(loc.meta_path.read_text())
        print(f"\n=== {loc.dir} ===")
        for k, v in meta.items():
            if k in ("fact_ids", "address_strs"):
                v = f"[{len(v)} entries]"
            print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--root", required=True)
    p_list.add_argument("--model", default=None)
    p_list.set_defaults(fn=cmd_list)
    p_info = sub.add_parser("info")
    p_info.add_argument("--root", required=True)
    p_info.add_argument("--model", default=None)
    p_info.set_defaults(fn=cmd_info)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
