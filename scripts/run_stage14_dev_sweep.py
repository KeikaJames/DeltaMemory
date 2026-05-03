"""Stage 14 dev sweep: v2 vs v3 capture policies on the LAMA-TREx dev split.

Runs on Gemma-4-E2B (Mac MPS bf16). For each fact in eval/splits/dev.jsonl
we measure target-token recall@1 averaged over the fact's paraphrase set
under five conditions:

    B0  no-memory      (model alone; alpha = 0)
    B5  v2 / period    (legacy capture at last token)
    v3  policy=address (Stage 14B)
    v3  policy=multi   (Stage 14C: address + period)
    v3  policy=multi + tau=0.5 (Stage 14C + 14D)

ROME (14E) and trained InfoNCE K-projector (14A) require a separate train
pass; this script reports only the zero-training, zero-tuning portion of
v3 so we can pick the *capture policy* from dev. ROME and projector
training run in subsequent scripts.

Output: reports/cleanroom/stage14_dev/REPORT.md with per-condition
recall@1 (mean across seeds, per-seed values, count of facts) and a
summary CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deltamemory.memory.attn_native_bank import (  # noqa: E402
    AttnNativePatcher,
    forward_with_bank,
    fresh_bank,
    write_fact,
)


@dataclass
class FactRecord:
    fact_id: str
    address: str
    value: str
    write_prompt: str
    paraphrases: list[str]


def _load_dev() -> list[FactRecord]:
    path = REPO_ROOT / "eval" / "splits" / "dev.jsonl"
    out: list[FactRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        addr = rec["address_canonical"]
        value = rec["value"]
        out.append(
            FactRecord(
                fact_id=f"{rec['relation']}::{rec['entity']}",
                address=addr,
                value=value,
                write_prompt=f"{addr} {value}.",
                paraphrases=list(rec["paraphrases"]),
            )
        )
    return out


def _target_first_token_id(tokenizer, value: str) -> int:
    # Target = first sub-token of " value" (leading space; matches LM head decoding).
    ids = tokenizer(" " + value, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    return int(ids[0])


def _recall_at_1_for_fact(
    patcher,
    bank,
    tokenizer,
    fact: FactRecord,
    alpha: float,
) -> float:
    """Mean recall@1 over the fact's paraphrase prompts."""
    target_id = _target_first_token_id(tokenizer, fact.value)
    hits = 0
    n = 0
    for p in fact.paraphrases:
        logits = forward_with_bank(patcher, bank, tokenizer, p, alpha=alpha)
        pred = int(logits.argmax().item())
        hits += int(pred == target_id)
        n += 1
    return hits / max(n, 1)


def _eval_condition(
    *,
    patcher,
    tokenizer,
    facts: list[FactRecord],
    seed: int,
    policy: str | None,
    bank_temperature: float,
    read_alpha: float,
) -> dict:
    """Build a fresh bank, write all facts under the given policy, score recall@1."""
    rng = random.Random(seed)
    order = list(range(len(facts)))
    rng.shuffle(order)

    bank = fresh_bank(patcher.model)
    bank.bank_temperature = bank_temperature

    if policy is not None:
        for i in order:
            f = facts[i]
            write_fact(
                patcher=patcher,
                bank=bank,
                tokenizer=tokenizer,
                write_prompt=f.write_prompt,
                fact_id=f.fact_id,
                address=f.address,
                policy=policy,
            )

    per_fact = []
    for f in facts:
        r = _recall_at_1_for_fact(patcher, bank, tokenizer, f, alpha=read_alpha)
        per_fact.append(r)
    mean = sum(per_fact) / max(len(per_fact), 1)
    return {
        "policy": policy,
        "bank_temperature": bank_temperature,
        "read_alpha": read_alpha,
        "seed": seed,
        "n_facts": len(facts),
        "recall_at_1_mean": mean,
        "per_fact": per_fact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--limit", type=int, default=0, help="0 = all dev facts")
    parser.add_argument("--out", default="reports/cleanroom/stage14_dev")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    facts = _load_dev()
    if args.limit > 0:
        facts = facts[: args.limit]
    print(f"[stage14-dev] {len(facts)} facts, seeds={seeds}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16
    print(f"[stage14-dev] loading {args.model} on {args.device} bf16…", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    print(f"[stage14-dev] model ready in {time.time() - t0:.1f}s", flush=True)

    patcher = AttnNativePatcher(model)

    conditions = [
        # name, policy, tau, read_alpha
        ("B0_no_memory",       None,       1.0, 0.0),
        ("B5_v2_period",      "period",   1.0, 1.0),
        ("v3_address",        "address",  1.0, 1.0),
        ("v3_multi",          "multi",    1.0, 1.0),
        ("v3_multi_tau05",    "multi",    0.5, 1.0),
    ]

    rows: list[dict] = []
    for name, policy, tau, ra in conditions:
        for seed in seeds:
            print(f"[stage14-dev] {name} seed={seed}…", flush=True)
            t0 = time.time()
            r = _eval_condition(
                patcher=patcher,
                tokenizer=tokenizer,
                facts=facts,
                seed=seed,
                policy=policy,
                bank_temperature=tau,
                read_alpha=ra,
            )
            r["condition"] = name
            r["elapsed_sec"] = time.time() - t0
            rows.append(r)
            print(
                f"[stage14-dev]   recall@1 = {r['recall_at_1_mean']:.4f}  "
                f"({r['elapsed_sec']:.1f}s)",
                flush=True,
            )

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )

    by_cond: dict[str, list[float]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r["recall_at_1_mean"])

    md = ["# Stage 14 dev sweep — Gemma-4-E2B / LAMA-TREx dev (33 facts)\n"]
    md.append("| condition | seed mean | per-seed recall@1 |")
    md.append("|---|---:|---|")
    for cond, vals in by_cond.items():
        m = sum(vals) / len(vals)
        per = ", ".join(f"{v:.3f}" for v in vals)
        md.append(f"| {cond} | {m:.3f} | {per} |")
    md.append("")
    md.append("Generated by `scripts/run_stage14_dev_sweep.py`.")
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[stage14-dev] wrote {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
