#!/usr/bin/env python3
"""Phase mHC-flagship: cross-model α-safety sweep with mHC spectral shield.

Industrial-grade reproduction of the v3.2 thesis: with the mHC spectral
shield ON, the *safe α range* for DeltaMemory bank injection collapses to
a single shared value across all flagship LLMs.  Without the shield, each
family needs its own per-architecture α (Gemma-4 ≈ 1.0, Qwen3 ≈ 0.05,
Llama/DeepSeek-Qwen2 ≈ 0.05, GLM-4 ≈ 0.05 — a 20× spread that v3.1 is
forced to hand-calibrate).

For each (model × shield ∈ {off, on} × α) cell we measure:

  1. Counter-prior lift  Δ logP(wrong-target) on FALSE_FACTS.  This is
     the "did the bank actually inject?" signal — large positive lift
     means the model contradicted its own prior because of the bank.
  2. Baseline NLL drift on a clean held-out prompt set (no fact written;
     bank empty but α applied via writing one neutral fact).  This is
     the "did the model stay sane?" signal — small NLL drift means the
     architecture survived high α without collapse.

Headline metric: SHIELD reduces the cross-family α spread of the
Pareto-optimal point (max lift, NLL drift ≤ 0.5 nats) by ≥ 5×.

Red-line audit:
  * No LLM weights are touched.  ``AttnNativePatcher`` only swaps a
    forward method; α=0 is bit-equal (covered by tests/test_mhc_shield.py).
  * mHC shield is a parameter-free deterministic projector
    (``deltamemory.memory.mhc_shield``).  No new trainable parameters.

Usage:
    .venv-mac/bin/python scripts/run_mhc_flagship_sweep.py \\
        --models google/gemma-4-E2B \\
        --alphas 0.1 0.5 1.0 2.0 5.0 \\
        --shield both \\
        --device mps --dtype bfloat16 \\
        --out reports/cleanroom/mhc_flagship_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher,
    fresh_bank,
    forward_with_bank,
    write_fact,
)
from scripts.run_intervention_demo import FACTS, FALSE_FACTS  # reuse


# -- Held-out neutral prompts for NLL drift measurement.  These are
# generic Wikipedia-style sentences, deliberately unrelated to any of the
# FACTS / FALSE_FACTS subjects, so the bank should NOT help and we are
# isolating the "did α destabilise the model" signal.
NEUTRAL_PROMPTS = [
    "The boiling point of water at sea level is one hundred degrees Celsius.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "The Pacific Ocean covers approximately one third of the Earth's surface area.",
    "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    "Mount Everest is the tallest mountain above sea level, located in the Himalayan range.",
    "The Industrial Revolution began in Britain during the late eighteenth century.",
    "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
]


def _autodevice() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_model(name: str, device: str, dtype: torch.dtype):
    print(f"[load] {name}  device={device}  dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


@torch.no_grad()
def _next_token_logprob(model, tok, prompt: str, target_first_tok_id: int) -> float:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=am, use_cache=False)
    last = am.sum(dim=1).item() - 1
    logits = out.logits[0, last].float()
    return F.log_softmax(logits, dim=-1)[target_first_tok_id].item()


@torch.no_grad()
def _patched_logprob(patcher, bank, tok, prompt: str, target_first_tok_id: int,
                     alpha: float) -> float:
    logits = forward_with_bank(patcher, bank, tok, prompt, alpha=alpha).float()
    return F.log_softmax(logits, dim=-1)[target_first_tok_id].item()


@torch.no_grad()
def _seq_nll(model, tok, prompt: str) -> float:
    """Mean per-token NLL of ``prompt`` under the (possibly patched) model.
    Lower is better; an exploded model gives huge NLL."""
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=am, use_cache=False)
    logits = out.logits[0]                       # [T, V]
    targets = ids[0, 1:]                         # next-token targets
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item()


@torch.no_grad()
def _seq_nll_patched(patcher, bank, tok, prompt: str, alpha: float) -> float:
    """Patched-forward variant of _seq_nll.  Uses the same bank/alpha as the
    surrounding context.  We reuse ``forward_with_bank`` semantics by
    running the forward via patcher's context manager directly."""
    device = next(patcher.model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with patcher.attached(bank=bank, alpha=alpha):
        out = patcher.model(input_ids=ids, attention_mask=am, use_cache=False)
    logits = out.logits[0]
    targets = ids[0, 1:]
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item()


def run_cell(model_name: str, tok, model, alphas, shield_modes, false_facts,
             neutral_prompts, out_dir: Path, seeds=(0,)):
    """Run all (shield × α × seed) cells for one model."""
    cell_results = []
    short = model_name.replace("/", "_")
    for shield_on in shield_modes:
        for alpha in alphas:
            for seed in seeds:
                torch.manual_seed(seed)
                lifts = []
                for f in false_facts:
                    patcher = AttnNativePatcher(model)
                    bank = fresh_bank(model)
                    bank.mhc_shield = bool(shield_on)
                    bank.mhc_iters = 3
                    write_fact(patcher, bank, tok,
                               write_prompt=f["write"],
                               fact_id=f["fact_id"],
                               address=f["subject"])
                    target_id = tok(f["target"], add_special_tokens=False)["input_ids"][0]
                    base_lp = _next_token_logprob(model, tok, f["read"], target_id)
                    inj_lp = _patched_logprob(patcher, bank, tok,
                                              f["read"], target_id, alpha=alpha)
                    lifts.append(inj_lp - base_lp)

                # NLL drift on neutral prompts (one bank injected, neutral context).
                # We pre-load a single neutral fact so the bank is non-empty;
                # the question is whether α destabilises forward in unrelated context.
                patcher = AttnNativePatcher(model)
                bank = fresh_bank(model)
                bank.mhc_shield = bool(shield_on)
                bank.mhc_iters = 3
                write_fact(patcher, bank, tok,
                           write_prompt="Fact: The Sun is a star at the centre of the Solar System.",
                           fact_id="neutral_anchor",
                           address="the Sun")
                base_nlls, inj_nlls = [], []
                for p in neutral_prompts:
                    base_nlls.append(_seq_nll(model, tok, p))
                    inj_nlls.append(_seq_nll_patched(patcher, bank, tok, p, alpha=alpha))
                drift = (sum(inj_nlls) - sum(base_nlls)) / max(len(base_nlls), 1)

                cell = dict(
                    model=model_name,
                    shield=bool(shield_on),
                    alpha=float(alpha),
                    seed=int(seed),
                    mean_lift=float(sum(lifts) / max(len(lifts), 1)),
                    lifts=[float(x) for x in lifts],
                    nll_drift=float(drift),
                    n_facts=len(false_facts),
                    n_neutral=len(neutral_prompts),
                )
                cell_results.append(cell)
                print(f"  [{short}] shield={shield_on} α={alpha:>4.2f} seed={seed}"
                      f"  lift={cell['mean_lift']:+.3f}  drift={cell['nll_drift']:+.3f}",
                      flush=True)

    # Save per-model JSON.
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{short}.json").open("w") as fh:
        json.dump(cell_results, fh, indent=2)
    return cell_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="HuggingFace model IDs to sweep over (frozen LLMs).")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--shield", choices=["off", "on", "both"], default="both")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default="reports/cleanroom/mhc_flagship_sweep")
    args = ap.parse_args()

    device = args.device or _autodevice()
    dtype = _dtype(args.dtype)
    if args.shield == "off":
        shield_modes = [False]
    elif args.shield == "on":
        shield_modes = [True]
    else:
        shield_modes = [False, True]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        cmdline=" ".join(sys.argv),
        device=device, dtype=args.dtype,
        models=args.models, alphas=args.alphas,
        shield_modes=[bool(s) for s in shield_modes],
        seeds=args.seeds,
        n_false_facts=len(FALSE_FACTS),
        n_neutral=len(NEUTRAL_PROMPTS),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    aggregate = []
    for m in args.models:
        try:
            tok, model = _load_model(m, device=device, dtype=dtype)
        except Exception as exc:
            print(f"[skip] {m}: {exc}", flush=True)
            continue
        cells = run_cell(m, tok, model, args.alphas, shield_modes,
                         FALSE_FACTS, NEUTRAL_PROMPTS, out_dir, seeds=args.seeds)
        aggregate.extend(cells)
        del model, tok
        if device == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "AGGREGATE.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\n[done] {len(aggregate)} cells written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
