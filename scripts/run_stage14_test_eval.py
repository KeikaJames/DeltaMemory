"""Phase G test-set evaluation — Gemma-4-E2B / FROZEN v3.

Loads the **held-out test split** (39 facts) and the frozen v3 config, then
runs every preregistered baseline under the same protocol used on dev.
This is a one-shot run; per the preregistration, the test split must not be
touched again with this code path.

Conditions:
    B0  no_memory                    — model alone, alpha=0
    B1  prompt_insertion             — fact prepended in the prompt context
    B2  rag_oracle_text              — gold fact surfaced via in-context "lookup"
                                       (matched-budget, no retriever noise)
    v2  period_no_kproj              — Stage 13 baseline, no projector
    v3  period_kproj  (FROZEN)       — Stage 14 frozen config

For each condition we report per-fact recall@1 (averaged over paraphrases),
seed-mean recall@1, and per-fact arrays for downstream Wilcoxon /
bootstrap analysis. Three seeds are recorded but the eval pipeline is
deterministic so they only differ in fact-ordering at write time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage14_dev_sweep import FactRecord, _target_first_token_id  # noqa: E402
from deltamemory.memory.attn_native_bank import (  # noqa: E402
    AttnNativePatcher,
    forward_with_bank,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.k_projector import KProjectorBank  # noqa: E402


def _load_test(split_path: Path | None = None) -> list[FactRecord]:
    path = split_path if split_path is not None else (REPO_ROOT / "eval" / "splits" / "test.jsonl")
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


def _recall_no_bank(model, tok, facts: list[FactRecord], device: str,
                    prompt_prefix=lambda f: "") -> list[float]:
    """No-bank baseline. ``prompt_prefix(f)`` may inject context."""
    per_fact = []
    for f in facts:
        tid = _target_first_token_id(tok, f.value)
        hits = 0
        n = 0
        for p in f.paraphrases:
            text = prompt_prefix(f) + p
            with torch.inference_mode():
                ids = tok(text, return_tensors="pt").to(device)
                logits = model(**ids).logits[0, -1, :]
            hits += int(int(logits.argmax().item()) == tid)
            n += 1
        per_fact.append(hits / max(n, 1))
    return per_fact


def _recall_with_bank(patcher, tok, facts, *, policy, k_projector, tau,
                      seed: int, bank_topk: int = 0,
                      bank_cosine: bool = False,
                      bank_separate_softmax: bool = False,
                      bank_merge_beta: float = 1.0) -> list[float]:
    import random
    rng = random.Random(seed)
    order = list(range(len(facts)))
    rng.shuffle(order)

    bank = fresh_bank(patcher.model)
    bank.bank_temperature = tau
    bank.bank_topk = bank_topk
    bank.bank_cosine = bank_cosine
    bank.bank_separate_softmax = bank_separate_softmax
    bank.bank_merge_beta = bank_merge_beta
    if k_projector is not None:
        bank.k_projector = k_projector

    for i in order:
        f = facts[i]
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tok,
            write_prompt=f.write_prompt, fact_id=f.fact_id,
            address=f.address, policy=policy,
        )
    per_fact = []
    for f in facts:
        tid = _target_first_token_id(tok, f.value)
        hits = 0
        n = 0
        for p in f.paraphrases:
            logits = forward_with_bank(patcher, bank, tok, p, alpha=1.0)
            hits += int(int(logits.argmax().item()) == tid)
            n += 1
        per_fact.append(hits / max(n, 1))
    return per_fact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projector",
                    default="reports/cleanroom/stage14_kproj/k_projector.pt")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="reports/cleanroom/stage14_test_gemma4_e2b")
    ap.add_argument("--split", default=None,
                    help="Path to eval split jsonl (default: eval/splits/test.jsonl)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    split_path = Path(args.split) if args.split else None
    facts = _load_test(split_path)
    print(f"[test-eval] N={len(facts)} test facts", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[test-eval] loading {args.model}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.to(args.device).eval()
    print(f"[test-eval] model ready in {time.time() - t0:.1f}s", flush=True)

    patcher = AttnNativePatcher(model)
    proj = KProjectorBank.load(args.projector)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {}

    # B0: no_memory
    name = "B0_no_memory"
    rows = []
    pf = _recall_no_bank(model, tok, facts, args.device)
    for s in seeds:
        rows.append({"seed": s, "recall_at_1_mean": sum(pf)/len(pf), "per_fact": pf})
        print(f"[test-eval] {name} seed={s} recall@1={rows[-1]['recall_at_1_mean']:.4f}", flush=True)
    summary[name] = rows

    # B1: prompt_insertion -- prepend "<address> <value>." then the paraphrase.
    name = "B1_prompt_insertion"
    rows = []
    def _b1_prefix(f): return f"{f.address} {f.value}.\n"
    pf = _recall_no_bank(model, tok, facts, args.device, prompt_prefix=_b1_prefix)
    for s in seeds:
        rows.append({"seed": s, "recall_at_1_mean": sum(pf)/len(pf), "per_fact": pf})
        print(f"[test-eval] {name} seed={s} recall@1={rows[-1]['recall_at_1_mean']:.4f}", flush=True)
    summary[name] = rows

    # B2: RAG-oracle (Q&A-style context with the gold fact)
    name = "B2_rag_oracle"
    rows = []
    def _b2_prefix(f): return f"Reference: {f.address} {f.value}.\nQuestion: "
    pf = _recall_no_bank(model, tok, facts, args.device, prompt_prefix=_b2_prefix)
    for s in seeds:
        rows.append({"seed": s, "recall_at_1_mean": sum(pf)/len(pf), "per_fact": pf})
        print(f"[test-eval] {name} seed={s} recall@1={rows[-1]['recall_at_1_mean']:.4f}", flush=True)
    summary[name] = rows

    # v2: period_no_kproj
    name = "v2_period_no_kproj"
    rows = []
    for s in seeds:
        pf = _recall_with_bank(patcher, tok, facts, policy="period",
                               k_projector=None, tau=1.0, seed=s)
        rows.append({"seed": s, "recall_at_1_mean": sum(pf)/len(pf), "per_fact": pf})
        print(f"[test-eval] {name} seed={s} recall@1={rows[-1]['recall_at_1_mean']:.4f}", flush=True)
    summary[name] = rows

    # v3 FROZEN
    name = "v3_period_kproj"
    rows = []
    for s in seeds:
        pf = _recall_with_bank(patcher, tok, facts, policy="period",
                               k_projector=proj, tau=1.0, seed=s)
        rows.append({"seed": s, "recall_at_1_mean": sum(pf)/len(pf), "per_fact": pf})
        print(f"[test-eval] {name} seed={s} recall@1={rows[-1]['recall_at_1_mean']:.4f}", flush=True)
    summary[name] = rows

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table
    md = ["# Phase G test eval — Gemma-4-E2B (FROZEN v3)\n",
          f"**Test split**: N={len(facts)} held-out facts (untouched until this run).",
          f"**Frozen config**: deltamemory/configs/v3_frozen.yaml\n",
          "## Recall@1 by condition\n",
          "| Condition | mean | per-seed |", "|---|---:|---|"]
    for name, rows in summary.items():
        m = sum(r["recall_at_1_mean"] for r in rows) / max(len(rows), 1)
        per = ", ".join(f"{r['recall_at_1_mean']:.4f}" for r in rows)
        md.append(f"| {name} | {m:.4f} | {per} |")
    md.append("\nGenerated by `scripts/run_stage14_test_eval.py`.\n")
    (out_dir / "REPORT.md").write_text("\n".join(md))
    print(f"[test-eval] wrote {out_dir/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
