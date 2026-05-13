"""Exp31 Φ1 — Build train/val/test splits for K-adapter training.

Splits CounterFact-1k 700/150/150 by fact_id (stratified by P-relation when
possible). The split is deterministic for seed=0 so that any rebuild gives
the identical assignment. Distractors (10k) are shared across splits as
bank-padding for N>150.

Outputs (under data/splits/):
    train.json      — 700 facts (with paraphrases)
    val.json        — 150 facts
    test.json       — 150 facts
    distractors.json — list of distractor fact_ids (for bank padding)
    manifest.json   — sha256 of source files, split counts, seed

Run:
    python3 experiments/atb_validation_v1/exp31_learned_k_adapter/build_splits.py
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CF_PATH = REPO / "experiments" / "datasets" / "counterfact_1k.jsonl"
DISTRACTORS_PATH = REPO / "experiments" / "X1_bank_scaling" / "distractors.jsonl"
OUT_DIR = Path(__file__).parent / "data" / "splits"

SEED = 0
TRAIN_N = 700
VAL_N = 150
TEST_N = 150


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CF_PATH.open() as f:
        facts = [json.loads(line) for line in f if line.strip()]
    assert len(facts) == 1000, f"expected 1000 facts, got {len(facts)}"

    # Stratify by relation P-code: bucket facts, shuffle within bucket,
    # round-robin into train/val/test until quotas filled.
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for f_ in facts:
        by_rel[f_["relation"]].append(f_)

    rng = random.Random(SEED)
    for rel in sorted(by_rel):
        rng.shuffle(by_rel[rel])

    relations = sorted(by_rel)
    rng.shuffle(relations)  # rotate relation order deterministically

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    quotas = {"train": TRAIN_N, "val": VAL_N, "test": TEST_N}
    buckets = {"train": train, "val": val, "test": test}

    # Round-robin: take 1 fact per relation in rotation, dropping into the
    # split with the largest remaining quota. Stops when all quotas filled.
    pointers = {rel: 0 for rel in relations}
    while sum(quotas.values()) > 0:
        any_drawn = False
        for rel in relations:
            if pointers[rel] >= len(by_rel[rel]):
                continue
            target = max(quotas, key=lambda k: quotas[k])
            if quotas[target] == 0:
                break
            buckets[target].append(by_rel[rel][pointers[rel]])
            pointers[rel] += 1
            quotas[target] -= 1
            any_drawn = True
            if sum(quotas.values()) == 0:
                break
        if not any_drawn:
            raise RuntimeError("ran out of facts before quotas filled")

    assert len(train) == TRAIN_N
    assert len(val) == VAL_N
    assert len(test) == TEST_N
    assert len({f_["id"] for f_ in train + val + test}) == TRAIN_N + VAL_N + TEST_N

    # Load distractors (used by eval as bank padding for N > 150)
    with DISTRACTORS_PATH.open() as f:
        distractors = [json.loads(line) for line in f if line.strip()]
    distractor_ids = [d["fact_id"] for d in distractors]

    # Per-split paraphrase summary
    def paraphrase_stats(facts_: list[dict]) -> dict:
        counts = [len(f_.get("paraphrase_prompts", [])) for f_ in facts_]
        return {"min": min(counts), "max": max(counts), "mean": sum(counts) / len(counts)}

    # Write splits
    for name, bucket in (("train", train), ("val", val), ("test", test)):
        (OUT_DIR / f"{name}.json").write_text(json.dumps(bucket, indent=2, ensure_ascii=False))
        print(f"  {name}: {len(bucket)} facts, paraphrases={paraphrase_stats(bucket)}")

    # Distractor list
    (OUT_DIR / "distractors.json").write_text(json.dumps(distractor_ids, indent=2))
    print(f"  distractors: {len(distractor_ids)} ids")

    # Manifest
    manifest = {
        "seed": SEED,
        "sources": {
            "counterfact": {"path": str(CF_PATH.relative_to(REPO)), "sha256": sha256_file(CF_PATH)},
            "distractors": {"path": str(DISTRACTORS_PATH.relative_to(REPO)), "sha256": sha256_file(DISTRACTORS_PATH)},
        },
        "split_sizes": {"train": TRAIN_N, "val": VAL_N, "test": TEST_N},
        "distractor_count": len(distractor_ids),
        "stratification": "by relation P-code, round-robin to largest remaining quota",
        "relation_count": len(relations),
        "fact_id_overlap": 0,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
