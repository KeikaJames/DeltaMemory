"""Build leave-one-relation-out (LORO) splits from lama_trex_full.jsonl.

For each Wikidata relation r in the dataset, write two JSONL files:
  - lama_trex_loro_{r}_train.jsonl  (all facts whose relation != r)
  - lama_trex_loro_{r}_holdout.jsonl (all facts whose relation == r)

Used by Stage 10F. The training file is fed to run_stage8.py via
``--lama-jsonl``; the holdout file is fed via
``--stage10-loro-add-jsonl`` so that, after training, the held-out
relation's facts are appended to the bank without further training and
evaluated zero-shot.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "scripts" / "data" / "lama_trex_full.jsonl"
PARA = REPO_ROOT / "scripts" / "data" / "lama_trex_paraphrase.jsonl"
OUT_DIR = REPO_ROOT / "scripts" / "data" / "loro_splits"


def load() -> list[dict]:
    """Inner-join lama_trex_full with paraphrase JSONL on address to get relations."""
    rows_full = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    rel_by_addr = {}
    for line in PARA.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rel_by_addr[r["address"]] = r["relation"]
    out = []
    for f in rows_full:
        rel = rel_by_addr.get(f["address"])
        if rel:
            out.append({**f, "relation": rel})
    return out


def main() -> None:
    rows = load()
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_rel[r["relation"]].append(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for rel, recs in by_rel.items():
        if len(recs) < 5:
            print(f"SKIP {rel}: only {len(recs)} facts (too few)")
            continue
        train = [r for r in rows if r["relation"] != rel]
        holdout = recs
        (OUT_DIR / f"loro_{rel}_train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in train) + "\n"
        )
        (OUT_DIR / f"loro_{rel}_holdout.jsonl").write_text(
            "\n".join(json.dumps(r) for r in holdout) + "\n"
        )
        summary[rel] = {"train": len(train), "holdout": len(holdout)}
    (OUT_DIR / "splits_index.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
