#!/usr/bin/env python3
"""X.2 cross-arch comparator: merge two cells.jsonl runs (different models),
emit per-arch verdicts + side-by-side condition tables.

Usage:
    python3 experiments/X2_contradictory/cross_arch.py \
        --cells runs/X2_full_v1_qwen3/cells.jsonl \
                runs/X2_full_v1_gemma4e2b/cells.jsonl \
        --labels qwen3-4b gemma-4-e2b \
        --out runs/X2_cross_arch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import aggregate, load_cells  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, type=Path)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if len(args.cells) != len(args.labels):
        raise SystemExit("cells / labels length mismatch")

    per_arch = {}
    for path, label in zip(args.cells, args.labels):
        if not path.exists():
            print(f"[X2][cross_arch] WARN: {path} missing, skip", flush=True)
            continue
        cells = load_cells(path)
        per_arch[label] = aggregate(cells)
        print(f"[X2][cross_arch] {label}: {len(cells)} cells", flush=True)

    # cross-arch verdicts: same hypothesis supported across all archs?
    cross = {}
    if per_arch:
        first = next(iter(per_arch))
        verdicts = list(per_arch[first]["verdicts"].keys())
        for h in verdicts:
            supps = {arch: per_arch[arch]["verdicts"][h].get("supported")
                     for arch in per_arch}
            cross[h] = {
                "per_arch_supported": supps,
                "supported_in_all": all(s is True for s in supps.values()),
                "supported_in_any": any(s is True for s in supps.values()),
                "consistent": (
                    len(set(s for s in supps.values() if s is not None)) <= 1
                ),
            }

    out = {
        "n_archs": len(per_arch),
        "archs": list(per_arch.keys()),
        "cross_arch_verdicts": cross,
        "per_arch": per_arch,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[X2][cross_arch] -> {args.out}", flush=True)
    print("\nCross-arch verdicts:")
    for h, v in cross.items():
        print(f"  {h}:")
        for arch, s in v["per_arch_supported"].items():
            print(f"    {arch}: supported={s}")
        print(f"    -> consistent={v['consistent']} "
              f"all={v['supported_in_all']} any={v['supported_in_any']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
