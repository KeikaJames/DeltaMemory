"""Exp35b — 00: Build N=10k corpus from azhx/counterfact.

Mixes counterfact_1k.jsonl (already in repo) with 9000 additional facts from
the full azhx/counterfact dataset (19,728 facts). Stratifies by relation.
Targets ≥30% subject-collision frac across the bank.

Independent paraphrases are NOT generated here (separate step 01) because they
require Ollama and the corpus build is purely deterministic.

Output: exp35b_memit_bank/data/corpus_10k.jsonl
        exp35b_memit_bank/data/splits/{train,val,test}.json
        exp35b_memit_bank/data/corpus_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)


def stable_id(case_id, subject, relation, target_new):
    h = hashlib.sha1(f"{case_id}|{subject}|{relation}|{target_new}".encode()).hexdigest()[:12]
    return f"cf2_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subject-collision-min", type=float, default=0.30)
    ap.add_argument("--out", default=str(DATA / "corpus_10k.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset

    print("[load] azhx/counterfact ...", flush=True)
    ds = load_dataset("azhx/counterfact")
    k = list(ds.keys())[0]
    print(f"[load] {len(ds[k])} facts", flush=True)

    rng = random.Random(args.seed)

    # Pull all facts in canonical schema
    all_facts = []
    seen = set()
    for item in ds[k]:
        rw = item["requested_rewrite"]
        subject = str(rw["subject"]).strip()
        relation = str(rw["relation_id"])
        target_true = str(rw["target_true"]["str"]).strip()
        target_new = str(rw["target_new"]["str"]).strip()
        prompt = str(rw["prompt"])
        if not subject or not target_true or not target_new or not prompt:
            continue
        paraphrases = [p for p in item.get("paraphrase_prompts", []) if p]
        if len(paraphrases) < 2:
            continue
        case_id = item["case_id"]
        fid = stable_id(case_id, subject, relation, target_new)
        if fid in seen:
            continue
        seen.add(fid)
        all_facts.append({
            "id": fid,
            "case_id": case_id,
            "subject": subject,
            "relation": relation,
            "target_true": target_true,
            "target_new": target_new,
            "prompt": prompt,
            "paraphrase_prompts": paraphrases[:2],
            "neighborhood_prompts": item.get("neighborhood_prompts", [])[:5],
        })

    print(f"[filter] kept {len(all_facts)} usable facts", flush=True)

    # Stratify by relation
    by_rel: dict = defaultdict(list)
    for f in all_facts:
        by_rel[f["relation"]].append(f)
    rel_sizes = sorted([(r, len(v)) for r, v in by_rel.items()], key=lambda x: -x[1])
    print(f"[stratify] {len(by_rel)} relations; top: {rel_sizes[:5]}", flush=True)

    # Sample proportionally to relation freq (capped per relation)
    n_target = args.n
    sampled = []
    n_rels = len(by_rel)
    # First, take counterfact_1k subjects so they are guaranteed included
    cf1k_subjs = set()
    cf1k_path = Path(__file__).resolve().parents[3] / "experiments" / "datasets" / "counterfact_1k.jsonl"
    if cf1k_path.exists():
        for line in open(cf1k_path):
            r = json.loads(line)
            cf1k_subjs.add((r["subject"], r["relation"], r["target_true"]))

    forced = []
    remaining_per_rel = {}
    for r, facts in by_rel.items():
        rng.shuffle(facts)
        f1k_in_rel = [f for f in facts if (f["subject"], f["relation"], f["target_true"]) in cf1k_subjs]
        forced.extend(f1k_in_rel)
        remaining_per_rel[r] = [f for f in facts if f not in f1k_in_rel]

    sampled.extend(forced)
    n_forced = len(forced)
    print(f"[seed] forced {n_forced} from counterfact_1k overlap", flush=True)

    # Fill remainder proportionally
    remaining_total = sum(len(v) for v in remaining_per_rel.values())
    need = n_target - n_forced
    for r, facts in remaining_per_rel.items():
        quota = round(need * len(facts) / max(1, remaining_total))
        sampled.extend(facts[:quota])
    # Trim/pad to exactly n_target
    rng.shuffle(sampled)
    if len(sampled) > n_target:
        # Keep forced first
        forced_ids = {f["id"] for f in forced}
        keep = [f for f in sampled if f["id"] in forced_ids]
        rest = [f for f in sampled if f["id"] not in forced_ids]
        keep.extend(rest[: n_target - len(keep)])
        sampled = keep
    elif len(sampled) < n_target:
        # add more arbitrarily
        chosen_ids = {f["id"] for f in sampled}
        for f in all_facts:
            if len(sampled) >= n_target:
                break
            if f["id"] not in chosen_ids:
                sampled.append(f)
                chosen_ids.add(f["id"])

    print(f"[sample] selected {len(sampled)} facts", flush=True)

    # subject-collision metric BEFORE splitting
    subj_counts = Counter(f["subject"] for f in sampled)
    collisions = sum(c for c in subj_counts.values() if c > 1)
    collision_frac = collisions / len(sampled)
    print(f"[collision] subject-collision frac = {collision_frac:.2%}", flush=True)

    # Splits: 7000/1500/1500 fact-id-disjoint; allow subject overlap.
    rng.shuffle(sampled)
    splits = {
        "train": sampled[:7000],
        "val": sampled[7000:8500],
        "test": sampled[8500:10000],
    }

    # Subject collision in test must be measurable
    test_subjects = set(f["subject"] for f in splits["test"])
    test_with_collision = sum(1 for f in splits["test"] if subj_counts[f["subject"]] > 1)
    test_collision_frac = test_with_collision / len(splits["test"])

    # write
    with open(args.out, "w") as fh:
        for f in sampled:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    (DATA / "splits").mkdir(exist_ok=True)
    for s, items in splits.items():
        json.dump(items, open(DATA / "splits" / f"{s}.json", "w"), ensure_ascii=False, indent=1)

    meta = {
        "n_total": len(sampled),
        "n_forced_from_counterfact_1k": n_forced,
        "n_relations": len(by_rel),
        "relation_distribution": dict(Counter(f["relation"] for f in sampled).most_common()),
        "subject_collision_frac_overall": collision_frac,
        "subject_collision_frac_test": test_collision_frac,
        "splits": {s: len(v) for s, v in splits.items()},
        "seed": args.seed,
        "source": "azhx/counterfact",
        "corpus_sha": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
    }
    json.dump(meta, open(DATA / "corpus_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))

    # Pre-registered subject-collision threshold check (per D2)
    if collision_frac < 0.30:
        print(f"\n[WARN] subject-collision frac {collision_frac:.2%} < 0.30 — may need to bias sampling", flush=True)


if __name__ == "__main__":
    main()
