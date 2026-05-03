"""Stage 13B: robustness benchmarks on the AttentionNative DeltaMemory bank.

Three benchmarks against the zero-shot AttnNativeBank from Stage 13A:
  13B-1  Paraphrase robustness   (gate >= 0.70)
  13B-2  Decoy curve at K in {0,10,50,100}  (no hard gate, diagnostic)
  13B-3  LORO leave-one-relation-out        (gate >= 0.50)

All three use only the K/V concat injection — no learnable params,
unless --train-projector is set, in which case 13B-4 trains a single
linear K-projector with InfoNCE on paraphrase pairs.

Honest framing rule: every gate that fails stays FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deltamemory.gemma.model_adapter import load_model_bundle  # noqa: E402
from deltamemory.memory.attn_native_bank import (  # noqa: E402
    AttnNativeBank, AttnNativePatcher, fresh_bank, write_fact, forward_with_bank,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    fact_id: str
    address: str          # canonical query, e.g. "The capital of France is"
    value: str            # "Paris"
    value_token_id: int   # tokenizer.encode(" Paris")[0]
    relation: str | None = None
    paraphrases_holdout: list[str] = field(default_factory=list)


def _single_token(tokenizer, word: str) -> int | None:
    ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    return ids[0] if ids else None


def load_paraphrase_facts(tokenizer, n: int, seed: int) -> list[Fact]:
    """Load N facts from the LAMA paraphrase split.

    Address (canonical) comes from the train file; held-out paraphrases come
    from the holdout file (different surface form, same address+value).
    """
    train_path = REPO_ROOT / "scripts/data/lama_stage11_train_paraphrase.jsonl"
    hold_path  = REPO_ROOT / "scripts/data/lama_stage11_holdout_paraphrase.jsonl"
    train = [json.loads(l) for l in train_path.open()]
    hold  = {json.loads(l)["address_canonical"]: json.loads(l)["paraphrases"]
             for l in hold_path.open()}
    rng = random.Random(seed)
    rng.shuffle(train)
    facts: list[Fact] = []
    for row in train:
        addr = row["address_canonical"]
        val = row["value"].strip()
        tid = _single_token(tokenizer, val)
        if tid is None:
            continue
        facts.append(Fact(
            fact_id=f"{row.get('relation','?')}::{row.get('entity','?')}",
            address=addr, value=val, value_token_id=tid,
            relation=row.get("relation"),
            paraphrases_holdout=hold.get(addr, []),
        ))
        if len(facts) >= n:
            break
    return facts


def load_loro_facts(tokenizer, relation: str) -> tuple[list[Fact], list[Fact]]:
    base = REPO_ROOT / "scripts/data/loro_splits"
    def _load(p):
        out = []
        for line in p.open():
            row = json.loads(line)
            tid = _single_token(tokenizer, row["value"])
            if tid is None:
                continue
            out.append(Fact(
                fact_id=f"{row['relation']}::{row['address'][:32]}",
                address=row["address"], value=row["value"].strip(),
                value_token_id=tid, relation=row["relation"],
            ))
        return out
    train = _load(base / f"loro_{relation}_train.jsonl")
    hold  = _load(base / f"loro_{relation}_holdout.jsonl")
    return train, hold


# ---------------------------------------------------------------------------
# Bank helpers
# ---------------------------------------------------------------------------

def write_facts_bulk(patcher: AttnNativePatcher, bank: AttnNativeBank,
                     tokenizer, facts: list[Fact]) -> None:
    """Write each fact: 'ADDRESS VALUE.' captured at last token."""
    t0 = time.time()
    for f in facts:
        prompt = f"{f.address} {f.value}."
        write_fact(patcher, bank, tokenizer,
                   write_prompt=prompt, fact_id=f.fact_id, address=f.address)
    print(f"[bank] wrote {len(facts)} facts in {time.time()-t0:.1f}s", flush=True)


def sub_bank(full: AttnNativeBank, indices: list[int]) -> AttnNativeBank:
    """Construct a lightweight view over a row-subset of `full`."""
    idx = torch.tensor(indices, dtype=torch.long, device=full.M_K[0].device)
    nb = AttnNativeBank(
        num_layers=full.num_layers, num_kv_heads=full.num_kv_heads,
        head_dim=full.head_dim, head_dims=list(full.head_dims),
        device=full.device, dtype=full.dtype,
    )
    nb.M_K = [full.M_K[l].index_select(0, idx).contiguous() for l in range(full.num_layers)]
    nb.M_V = [full.M_V[l].index_select(0, idx).contiguous() for l in range(full.num_layers)]
    nb.fact_ids = [full.fact_ids[i] for i in indices]
    nb.address_strs = [full.address_strs[i] for i in indices]
    return nb


def predict_top1(patcher, bank, tokenizer, prompt: str, alpha: float) -> int:
    logits = forward_with_bank(patcher, bank, tokenizer, prompt, alpha=alpha).float()
    return int(logits.argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_paraphrase(patcher, tokenizer, facts: list[Fact],
                     alphas: list[float], n_para: int) -> dict:
    bank = fresh_bank(patcher.model)
    write_facts_bulk(patcher, bank, tokenizer, facts)
    out: dict = {"per_alpha": {}, "n_facts": len(facts), "n_para": n_para}
    for alpha in alphas:
        # Canonical query (sanity)
        canon_correct = 0
        para_correct = 0
        para_total = 0
        for f in facts:
            top1 = predict_top1(patcher, bank, tokenizer, f.address, alpha)
            if top1 == f.value_token_id:
                canon_correct += 1
            for q in f.paraphrases_holdout[:n_para]:
                t = predict_top1(patcher, bank, tokenizer, q, alpha)
                para_total += 1
                if t == f.value_token_id:
                    para_correct += 1
        out["per_alpha"][f"{alpha}"] = {
            "canonical_recall_at_1": canon_correct / max(1, len(facts)),
            "paraphrase_recall_at_1": para_correct / max(1, para_total),
            "para_total": para_total,
        }
        print(f"[13B-1 alpha={alpha}] canon={out['per_alpha'][f'{alpha}']['canonical_recall_at_1']:.3f}  "
              f"para={out['per_alpha'][f'{alpha}']['paraphrase_recall_at_1']:.3f}", flush=True)
    return out


def bench_decoy(patcher, tokenizer, facts: list[Fact],
                alphas: list[float], k_values: list[int],
                n_targets: int, seed: int) -> dict:
    """Decoy curve: for each K, measure recall using bank of (target + K decoys)."""
    bank = fresh_bank(patcher.model)
    write_facts_bulk(patcher, bank, tokenizer, facts)
    rng = random.Random(seed + 1000)
    target_indices = list(range(len(facts)))
    rng.shuffle(target_indices)
    target_indices = target_indices[:n_targets]
    out: dict = {"per_alpha": {}, "n_targets": n_targets}
    for alpha in alphas:
        per_k: dict = {}
        for K in k_values:
            correct = 0
            for ti in target_indices:
                pool = [j for j in range(len(facts)) if j != ti]
                rng.shuffle(pool)
                idx = [ti] + pool[:K]
                sb = sub_bank(bank, idx)
                t = predict_top1(patcher, sb, tokenizer, facts[ti].address, alpha)
                if t == facts[ti].value_token_id:
                    correct += 1
            per_k[f"K={K}"] = correct / max(1, n_targets)
        out["per_alpha"][f"{alpha}"] = per_k
        print(f"[13B-2 alpha={alpha}] " +
              "  ".join(f"K={K}:{per_k[f'K={K}']:.2f}" for K in k_values), flush=True)
    return out


def bench_loro(patcher, tokenizer, relations: list[str],
               alphas: list[float]) -> dict:
    out: dict = {"per_alpha": {}, "relations": relations}
    per_alpha: dict = {a: {"correct": 0, "total": 0, "splits": {}} for a in alphas}
    for rel in relations:
        train, hold = load_loro_facts(tokenizer, rel)
        if not hold:
            continue
        bank = fresh_bank(patcher.model)
        write_facts_bulk(patcher, bank, tokenizer, train)
        for alpha in alphas:
            c = 0
            for f in hold:
                t = predict_top1(patcher, bank, tokenizer, f.address, alpha)
                if t == f.value_token_id:
                    c += 1
            per_alpha[alpha]["correct"] += c
            per_alpha[alpha]["total"] += len(hold)
            per_alpha[alpha]["splits"][rel] = {"recall_at_1": c / len(hold), "n_hold": len(hold)}
            print(f"[13B-3 rel={rel} alpha={alpha}] hold n={len(hold)}  "
                  f"recall={c/len(hold):.3f}", flush=True)
        # Free bank tensors
        del bank
    for a in alphas:
        d = per_alpha[a]
        out["per_alpha"][f"{a}"] = {
            "macro_recall_at_1": (sum(s["recall_at_1"] for s in d["splits"].values())
                                  / max(1, len(d["splits"]))),
            "micro_recall_at_1": d["correct"] / max(1, d["total"]),
            "splits": d["splits"],
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-facts", type=int, default=100)
    ap.add_argument("--n-para", type=int, default=3)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alphas", default="0.5,1.0,1.5")
    ap.add_argument("--ks", default="0,10,50,100")
    ap.add_argument("--loro-relations", default="P36,P101,P19")
    ap.add_argument("--out-dir", default="reports/cleanroom/stage13b_robust")
    ap.add_argument("--skip", default="", help="comma-list of {paraphrase,decoy,loro}")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    alphas = [float(a) for a in args.alphas.split(",") if a]
    ks = [int(k) for k in args.ks.split(",") if k]
    relations = [r for r in args.loro_relations.split(",") if r]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print(f"[stage13b] loading {args.model} on {args.device} {args.dtype}", flush=True)
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype,
                               attn_implementation="eager")
    patcher = AttnNativePatcher(bundle.model)
    tok = bundle.tokenizer

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "config": vars(args),
        "seeds": seeds, "alphas": alphas,
        "model": args.model, "device": str(bundle.device), "dtype": str(bundle.dtype),
        "results": {"paraphrase": {}, "decoy": {}, "loro": {}},
    }

    # ---- 13B-1 paraphrase ----
    if "paraphrase" not in skip:
        print("\n=== 13B-1 paraphrase robustness ===", flush=True)
        for s in seeds:
            facts = load_paraphrase_facts(tok, n=args.n_facts, seed=s)
            print(f"[13B-1 seed={s}] facts={len(facts)} (req={args.n_facts})", flush=True)
            r = bench_paraphrase(patcher, tok, facts, alphas, n_para=args.n_para)
            summary["results"]["paraphrase"][f"seed={s}"] = r

    # ---- 13B-2 decoy curve ----
    if "decoy" not in skip:
        print("\n=== 13B-2 decoy curve ===", flush=True)
        for s in seeds:
            facts = load_paraphrase_facts(tok, n=args.n_facts, seed=s)
            r = bench_decoy(patcher, tok, facts, alphas,
                            k_values=ks, n_targets=args.n_targets, seed=s)
            summary["results"]["decoy"][f"seed={s}"] = r

    # ---- 13B-3 LORO ----
    if "loro" not in skip:
        print("\n=== 13B-3 LORO ===", flush=True)
        # LORO splits are deterministic; use one pass + replicate alpha across seeds.
        # We treat seeds as redundant here but record once for transparency.
        r = bench_loro(patcher, tok, relations, alphas)
        summary["results"]["loro"]["splits"] = r

    # ---- aggregate gate numbers ----
    gates: dict = {}
    if summary["results"]["paraphrase"]:
        # Best alpha across seeds, mean across seeds at each alpha
        per_a: dict = {a: [] for a in [str(x) for x in alphas]}
        for s in seeds:
            for a in [str(x) for x in alphas]:
                per_a[a].append(summary["results"]["paraphrase"][f"seed={s}"]["per_alpha"][a]
                                ["paraphrase_recall_at_1"])
        para_mean = {a: mean(v) for a, v in per_a.items()}
        best_alpha = max(para_mean, key=para_mean.get)
        gates["13B-1_paraphrase"] = {
            "metric": "paraphrase_recall_at_1",
            "per_alpha_mean_over_seeds": para_mean,
            "best_alpha": best_alpha,
            "best_value": para_mean[best_alpha],
            "gate_threshold": 0.70,
            "pass": para_mean[best_alpha] >= 0.70,
        }

    if summary["results"]["decoy"]:
        per_a: dict = {a: {f"K={k}": [] for k in ks} for a in [str(x) for x in alphas]}
        for s in seeds:
            r = summary["results"]["decoy"][f"seed={s}"]
            for a in [str(x) for x in alphas]:
                for k in ks:
                    per_a[a][f"K={k}"].append(r["per_alpha"][a][f"K={k}"])
        gates["13B-2_decoy"] = {
            "metric": "recall_at_1_vs_K",
            "per_alpha_mean_over_seeds": {
                a: {k: mean(v) for k, v in d.items()} for a, d in per_a.items()
            },
            "gate_threshold": None,
            "pass": True,  # diagnostic only
        }

    if summary["results"]["loro"]:
        loro = summary["results"]["loro"]["splits"]["per_alpha"]
        per_a_macro = {a: loro[a]["macro_recall_at_1"] for a in [str(x) for x in alphas]}
        best = max(per_a_macro, key=per_a_macro.get)
        gates["13B-3_loro"] = {
            "metric": "macro_recall_at_1",
            "per_alpha_macro": per_a_macro,
            "best_alpha": best,
            "best_value": per_a_macro[best],
            "gate_threshold": 0.50,
            "pass": per_a_macro[best] >= 0.50,
        }

    summary["gates"] = gates
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report_md(out_dir / "report.md", summary)
    print("\n=== SUMMARY ===")
    for name, g in gates.items():
        print(f"  {name}: best={g.get('best_value', 'n/a')} "
              f"thr={g.get('gate_threshold')}  pass={g['pass']}")


def write_report_md(path: Path, summary: dict) -> None:
    lines = [
        "# Stage 13B robustness benchmarks — AttentionNative DeltaMemory bank\n",
        f"- model: `{summary['model']}`",
        f"- device/dtype: `{summary['device']}` / `{summary['dtype']}`",
        f"- seeds: {summary['seeds']}    alphas: {summary['alphas']}",
        f"- N facts: {summary['config']['n_facts']}    paraphrases/fact: {summary['config']['n_para']}    decoy targets: {summary['config']['n_targets']}",
        f"- LORO relations: {summary['config']['loro_relations']}",
        "",
        "## Gates",
        "",
        "| Gate | metric | best | threshold | pass |",
        "|---|---|---|---|---|",
    ]
    for name, g in summary.get("gates", {}).items():
        bv = g.get("best_value", "")
        bv_s = f"{bv:.3f}" if isinstance(bv, float) else "(diagnostic)"
        thr = g.get("gate_threshold")
        thr_s = "n/a" if thr is None else f"{thr}"
        lines.append(f"| {name} | {g['metric']} | {bv_s} | {thr_s} | {'PASS' if g['pass'] else 'FAIL'} |")
    lines.append("")
    # 13B-1 detail
    if "13B-1_paraphrase" in summary.get("gates", {}):
        g = summary["gates"]["13B-1_paraphrase"]
        lines += ["### 13B-1 paraphrase recall@1 (mean across seeds)", ""]
        for a, v in g["per_alpha_mean_over_seeds"].items():
            lines.append(f"- alpha={a}: {v:.3f}")
        lines.append("")
    # 13B-2 detail
    if "13B-2_decoy" in summary.get("gates", {}):
        g = summary["gates"]["13B-2_decoy"]
        lines += ["### 13B-2 decoy curve recall@1 (mean across seeds)", ""]
        for a, d in g["per_alpha_mean_over_seeds"].items():
            row = "  ".join(f"{k}: {v:.3f}" for k, v in d.items())
            lines.append(f"- alpha={a}: {row}")
        lines.append("")
    # 13B-3 detail
    if "13B-3_loro" in summary.get("gates", {}):
        g = summary["gates"]["13B-3_loro"]
        lines += ["### 13B-3 LORO macro recall@1", ""]
        for a, v in g["per_alpha_macro"].items():
            lines.append(f"- alpha={a}: {v:.3f}")
        per_split = summary["results"]["loro"]["splits"]["per_alpha"][g["best_alpha"]]["splits"]
        lines.append("")
        lines.append("Per-split (best alpha):")
        for rel, d in per_split.items():
            lines.append(f"- {rel}: recall@1={d['recall_at_1']:.3f}  (n_hold={d['n_hold']})")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
