"""Stratified train/dev/test split for the LAMA-TREx paraphrase set.

Run once. Output is committed to the repo and SHA-256 hashed into
`docs/preregistration.md` so that any post-hoc tweak to the test set is
detectable.

Determinism contract:
  - Seed = 42
  - Stratification key = relation (P36, P101, ...)
  - Split ratio = 60% train / 20% dev / 20% test (rounded down per stratum)
  - Sort order: by (relation, entity) ASCII before shuffle so that the
    pre-shuffle ordering does not depend on filesystem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "scripts" / "data" / "lama_trex_paraphrase.jsonl"
OUT_DIR = REPO_ROOT / "eval" / "splits"
SEED = 42
RATIOS = (0.60, 0.20, 0.20)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def split() -> dict:
    rng = random.Random(SEED)
    records = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_rel[rec["relation"]].append(rec)

    train, dev, test = [], [], []
    for rel in sorted(by_rel):
        bucket = sorted(by_rel[rel], key=lambda r: (r["relation"], r["entity"]))
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * RATIOS[0])
        n_dev = int(n * RATIOS[1])
        train += bucket[:n_train]
        dev += bucket[n_train : n_train + n_dev]
        test += bucket[n_train + n_dev :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train": OUT_DIR / "train.jsonl",
        "dev": OUT_DIR / "dev.jsonl",
        "test": OUT_DIR / "test.jsonl",
    }
    _write_jsonl(out_paths["train"], train)
    _write_jsonl(out_paths["dev"], dev)
    _write_jsonl(out_paths["test"], test)

    manifest = {
        "seed": SEED,
        "ratios": list(RATIOS),
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": _sha256_file(SOURCE),
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "sha256": {name: _sha256_file(p) for name, p in out_paths.items()},
        "stratification": "relation",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Re-run and verify SHAs match manifest.")
    args = parser.parse_args()

    if args.check:
        existing = json.loads((OUT_DIR / "manifest.json").read_text())
        fresh = split()
        assert existing == fresh, "Holdout split drifted; investigate before any test-set evaluation."
        print("OK: holdout split deterministic and matches committed manifest.")
        return

    manifest = split()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
