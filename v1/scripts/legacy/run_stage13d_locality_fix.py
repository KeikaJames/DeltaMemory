#!/usr/bin/env python3
"""Stage 13D — per-query routing fix for P3 locality drift.

Stage 12 P3 measured drift = 0.75 because the v1 broadcast pipeline injects
``alpha * mean(bank)`` at EVERY query's last hidden state, regardless of
whether the query has anything to do with the bank. This script implements a
per-query cosine gate on the v1 read pipeline (NOT v2 attn_native_bank.py):

    q_key = mean-pool(input_embed(query_tokens))   # parameter-free
    k_i   = mean-pool(input_embed(address_i))      # parameter-free
    cos_max = max_i cos(q_key, k_i)
    top1    = argmax_i cos(q_key, k_i)
    alpha_eff = alpha * sigmoid(beta*(cos_max - tau))     # soft
              ≈ alpha if cos_max>=tau else 0              # hard
    h <- h + alpha_eff * bank[top1]

For the persistent-bank vector we use ``lm_head.weight[value_token_id]`` so
that override is deterministic and we don't need to train a Writer (which
would blow the 30-min budget on MPS). This isolates the gating fix from any
optimization noise: a successful override means the gate routed correctly.

Outputs:
    reports/cleanroom/stage13d_locality_fix/report.md
    reports/cleanroom/stage13d_locality_fix/summary.json

Pass criterion: drift <= 0.05 with override >= 0.90 at some tau.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_stage8 import build_facts_lama, _tokenize_read_prompts, Fact  # noqa: E402


# -----------------------------------------------------------------------------
# 50 unrelated control prompts (NOT present in LAMA bank). Hand-curated so
# none of them share an "address" string with any fact in the bank.
# -----------------------------------------------------------------------------
CONTROL_PROMPTS: list[str] = [
    "Two plus two equals",
    "A triangle has three",
    "Photosynthesis converts sunlight into",
    "The opposite of hot is",
    "The speed of light is approximately",
    "Water freezes at zero degrees",
    "Humans typically have ten",
    "An octopus has eight",
    "The sun rises in the",
    "A year on Earth has twelve",
    "The smallest prime number is",
    "Bees produce a sweet substance called",
    "Fish breathe through their",
    "A baby dog is called a",
    "Snow is typically the color",
    "The first letter of the alphabet is",
    "Five times five is",
    "Ice is frozen",
    "Birds lay",
    "Spiders have eight",
    "An hour has sixty",
    "A minute has sixty",
    "A week has seven",
    "The largest planet in our solar system is",
    "The moon orbits the",
    "Lightning is followed by",
    "Plants need water and",
    "A square has four",
    "The sky on a clear day is",
    "Rain falls from the",
    "Mammals are warm",
    "Trees are made of",
    "Salt makes food taste",
    "Sugar tastes",
    "Fire is",
    "Diamonds are very",
    "An apple a day keeps the doctor",
    "Time flies like an",
    "Practice makes",
    "Better late than",
    "Actions speak louder than",
    "Birds of a feather flock",
    "Don't judge a book by its",
    "A picture is worth a thousand",
    "When in Rome, do as the Romans",
    "The early bird catches the",
    "Beauty is in the eye of the",
    "Curiosity killed the",
    "All that glitters is not",
    "Where there's smoke, there's",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def mean_pool_embed(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the (frozen) input-token embeddings → (B, H), float."""
    embed = model.get_input_embeddings()
    e = embed(ids)
    m = mask.unsqueeze(-1).float()
    pooled = (e.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return pooled  # (B, H)


def tokenize_no_special(tokenizer, texts: list[str], device, max_len: int = 64):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                    add_special_tokens=False, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def forward_read_with_perquery_injection(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    last_pos: torch.Tensor,
    injection_vectors: torch.Tensor | None,
    alpha_eff: torch.Tensor | float,
) -> torch.Tensor:
    """Like _forward_read_with_injection but alpha_eff may be per-query.

    injection_vectors: (B, H) or None
    alpha_eff: scalar OR (B,) tensor.
    """
    out = model.model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=False, use_cache=False,
    )
    last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    B, L, H = last_hidden.shape
    idx = last_pos.view(B, 1, 1).expand(B, 1, H)
    h = last_hidden.gather(1, idx).squeeze(1)
    if injection_vectors is not None:
        if isinstance(alpha_eff, torch.Tensor):
            a = alpha_eff.to(h.dtype).view(B, 1)
        else:
            a = float(alpha_eff)
        h = h + a * injection_vectors.to(h.dtype)
    return model.lm_head(h)


# -----------------------------------------------------------------------------
# Probes
# -----------------------------------------------------------------------------
@dataclass
class GateConfig:
    enabled: bool
    tau: float = 0.6
    beta: float = 20.0
    soft: bool = True


def per_query_route(
    q_keys: torch.Tensor,    # (Q, H) float
    bank_keys: torch.Tensor, # (N, H) float
    bank_vecs: torch.Tensor, # (N, H_model)
    alpha: float,
    gate: GateConfig,
):
    """Return (top1_slot, cos_max, alpha_eff_per_query, injection_vectors)."""
    qn = F.normalize(q_keys, dim=-1)
    kn = F.normalize(bank_keys, dim=-1)
    sims = qn @ kn.t()                  # (Q, N)
    cos_max, top1 = sims.max(dim=-1)    # (Q,), (Q,)
    Q = q_keys.shape[0]
    if not gate.enabled:
        alpha_eff = torch.full((Q,), float(alpha), device=q_keys.device)
    elif gate.soft:
        alpha_eff = alpha * torch.sigmoid(gate.beta * (cos_max - gate.tau))
    else:
        alpha_eff = torch.where(cos_max >= gate.tau,
                                torch.full_like(cos_max, alpha),
                                torch.zeros_like(cos_max))
    inject = bank_vecs[top1]            # (Q, H)
    return top1, cos_max, alpha_eff, inject


def measure_p3(
    model, tokenizer, facts: list[Fact],
    bank_keys: torch.Tensor, bank_vecs: torch.Tensor,
    controls: list[str], alpha: float, gate: GateConfig, device,
):
    """Run override (canonical fact prompts) and locality drift (controls)."""
    # --- override on canonical prompts ---
    canon_prompts = [f.read_prompt for f in facts]
    value_ids = torch.tensor([f.value_token_id for f in facts], device=device)
    ids, am, lp = _tokenize_read_prompts(tokenizer, canon_prompts, device)

    # Per-query keys for canonical prompts: mean-pool of FULL prompt embeds.
    # (Equivalent intent: "what does this query semantically point to?")
    with torch.no_grad():
        q_keys = mean_pool_embed(model, ids, am)
    top1_c, cos_c, alpha_c, inject_c = per_query_route(
        q_keys, bank_keys, bank_vecs, alpha, gate)

    with torch.no_grad():
        base_logits = forward_read_with_perquery_injection(
            model, ids, am, lp, None, 0.0)
        base_argmax = base_logits.argmax(dim=-1)
        base_top1 = (base_argmax == value_ids).float().mean().item()

        dm_logits = forward_read_with_perquery_injection(
            model, ids, am, lp, inject_c, alpha_c)
        dm_argmax = dm_logits.argmax(dim=-1)
        dm_top1 = (dm_argmax == value_ids).float().mean().item()

        wrong_mask = (base_argmax != value_ids)
        if wrong_mask.sum() > 0:
            override = ((dm_argmax == value_ids) & wrong_mask).float().sum().item() / wrong_mask.sum().item()
        else:
            override = float('nan')

        # Routing correctness on canonical: top1 should match own slot
        gold_slot = torch.arange(len(facts), device=device)
        route_acc = (top1_c == gold_slot).float().mean().item()

    # --- locality drift on controls ---
    c_ids, c_am, c_lp = _tokenize_read_prompts(tokenizer, controls, device)
    with torch.no_grad():
        cq_keys = mean_pool_embed(model, c_ids, c_am)
    top1_u, cos_u, alpha_u, inject_u = per_query_route(
        cq_keys, bank_keys, bank_vecs, alpha, gate)

    with torch.no_grad():
        c_base_logits = forward_read_with_perquery_injection(
            model, c_ids, c_am, c_lp, None, 0.0)
        c_dm_logits = forward_read_with_perquery_injection(
            model, c_ids, c_am, c_lp, inject_u, alpha_u)
        c_drift = (c_base_logits.argmax(dim=-1) != c_dm_logits.argmax(dim=-1)).float().mean().item()

    return {
        "base_top1": base_top1,
        "DM_top1": dm_top1,
        "override_rate_on_wrong": override,
        "n_base_wrong": int(wrong_mask.sum().item()),
        "locality_drift_rate": c_drift,
        "route_acc_canonical": route_acc,
        "cos_max_canonical_mean": float(cos_c.mean().item()),
        "cos_max_controls_mean": float(cos_u.mean().item()),
        "alpha_eff_canonical_mean": float(alpha_c.mean().item()),
        "alpha_eff_controls_mean": float(alpha_u.mean().item()),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=50.0,
                    help="Scaling for lm_head-row injection. With "
                         "bank=lm_head.weight[v], the boost on the gold logit "
                         "is alpha*||lm_head_v||^2; alpha=50 reliably forces "
                         "override under bf16 on gemma-4-E2B.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-dir", default="reports/cleanroom/stage13d_locality_fix")
    ap.add_argument("--lama-jsonl", default="scripts/data/lama_curated.jsonl")
    ap.add_argument("--taus", type=str, default="0.3,0.5,0.6,0.7,0.72,0.75")
    ap.add_argument("--beta", type=float, default=40.0)
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- load model ----------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[s13d] loading google/gemma-4-E2B", flush=True)
    t0 = time.time()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B", torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[s13d] loaded in {time.time()-t0:.1f}s", flush=True)

    # ---------------- build facts ----------------
    facts = build_facts_lama(tok, REPO_ROOT / args.lama_jsonl, seed=args.seed,
                             n_facts=args.n_facts)
    n = len(facts)
    print(f"[s13d] N facts = {n}", flush=True)

    # Filter controls so none of them clash with a bank address (case-insensitive
    # substring overlap).
    bank_addrs_lower = [f.address.lower() for f in facts]
    controls = [c for c in CONTROL_PROMPTS
                if not any(c.lower() == a or c.lower() in a or a in c.lower()
                           for a in bank_addrs_lower)]
    print(f"[s13d] N controls = {len(controls)}", flush=True)

    device = args.device

    # ---------------- bank: keys + vectors ----------------
    addrs = [f.address for f in facts]
    addr_ids, addr_mask = tokenize_no_special(tok, addrs, device)
    with torch.no_grad():
        bank_keys = mean_pool_embed(model, addr_ids, addr_mask)  # (N, H)
        # Bank vectors: lm_head row for each gold value token. Adding
        # alpha*lm_head_row to h adds alpha*||lm_head_row||^2 to that token's
        # logit, deterministically driving override.
        value_ids = torch.tensor([f.value_token_id for f in facts], device=device)
        lmh = model.get_output_embeddings().weight  # (V, H)
        bank_vecs = lmh.index_select(0, value_ids).contiguous()  # (N, H)

    # ---------------- baseline (NO gating, broadcast mean(bank)) ----------------
    # This matches Stage 12's drift measurement exactly: alpha * mean(bank) on
    # every control. Reused per-query route with gate disabled gives equivalent
    # behaviour for canonicals; for the broadcast baseline we explicitly inject
    # mean(bank) on controls.
    print("[s13d] === baseline (no gating, broadcast mean(bank)) ===", flush=True)
    canon_prompts = [f.read_prompt for f in facts]
    ids, am, lp = _tokenize_read_prompts(tok, canon_prompts, device)
    with torch.no_grad():
        # Override with own-slot injection at full alpha (best case for v1).
        base_logits = forward_read_with_perquery_injection(model, ids, am, lp, None, 0.0)
        base_argmax = base_logits.argmax(dim=-1)
        base_top1 = (base_argmax == value_ids).float().mean().item()
        dm_logits = forward_read_with_perquery_injection(
            model, ids, am, lp, bank_vecs, args.alpha)
        dm_argmax = dm_logits.argmax(dim=-1)
        dm_top1 = (dm_argmax == value_ids).float().mean().item()
        wrong_mask = (base_argmax != value_ids)
        if wrong_mask.sum() > 0:
            base_override = ((dm_argmax == value_ids) & wrong_mask).float().sum().item() / wrong_mask.sum().item()
        else:
            base_override = float('nan')

        # Drift: broadcast mean(bank) to all controls (the v1 bug).
        c_ids, c_am, c_lp = _tokenize_read_prompts(tok, controls, device)
        mean_bank = bank_vecs.mean(dim=0, keepdim=True).expand(len(controls), -1)
        c_base = forward_read_with_perquery_injection(model, c_ids, c_am, c_lp, None, 0.0)
        c_dm = forward_read_with_perquery_injection(
            model, c_ids, c_am, c_lp, mean_bank, args.alpha)
        base_drift = (c_base.argmax(dim=-1) != c_dm.argmax(dim=-1)).float().mean().item()

    baseline = {
        "alpha": args.alpha,
        "base_top1": base_top1,
        "DM_top1": dm_top1,
        "override_rate_on_wrong": base_override,
        "locality_drift_rate": base_drift,
        "n_base_wrong": int(wrong_mask.sum().item()),
    }
    print(f"[s13d] baseline: drift={base_drift:.3f} override={base_override:.3f}",
          flush=True)

    # ---------------- gated sweep over tau ----------------
    taus = [float(x) for x in args.taus.split(",")]
    sweep = []
    for tau in taus:
        for soft, label in [(True, "soft"), (False, "hard")]:
            gate = GateConfig(enabled=True, tau=tau, beta=args.beta, soft=soft)
            print(f"[s13d] gate tau={tau} {label} ...", flush=True)
            res = measure_p3(model, tok, facts, bank_keys, bank_vecs,
                             controls, args.alpha, gate, device)
            res["tau"] = tau
            res["beta"] = args.beta
            res["mode"] = label
            sweep.append(res)
            print(f"[s13d]   drift={res['locality_drift_rate']:.3f} "
                  f"override={res['override_rate_on_wrong']:.3f} "
                  f"route_acc={res['route_acc_canonical']:.3f} "
                  f"cos_canon={res['cos_max_canonical_mean']:.3f} "
                  f"cos_ctrl={res['cos_max_controls_mean']:.3f}",
                  flush=True)

    # ---------------- pass criterion ----------------
    passed = [r for r in sweep
              if r["locality_drift_rate"] <= 0.05
              and (r["override_rate_on_wrong"] >= 0.90
                   or (isinstance(r["override_rate_on_wrong"], float)
                       and math.isnan(r["override_rate_on_wrong"])))]
    # Strict: require explicit >=0.90, not nan.
    passed_strict = [r for r in sweep
                     if r["locality_drift_rate"] <= 0.05
                     and isinstance(r["override_rate_on_wrong"], float)
                     and not math.isnan(r["override_rate_on_wrong"])
                     and r["override_rate_on_wrong"] >= 0.90]
    status = "PASS" if passed_strict else "FAIL"

    summary = {
        "stage": "13D_locality_fix",
        "status": status,
        "n_facts": n,
        "n_controls": len(controls),
        "alpha": args.alpha,
        "model": "google/gemma-4-E2B",
        "device": args.device,
        "dtype": args.dtype,
        "baseline_no_gating": baseline,
        "gated_sweep": sweep,
        "passing_configs": passed_strict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- report ----------------
    lines = []
    lines.append(f"# Stage 13D — Per-query routing fix for P3 locality drift")
    lines.append("")
    lines.append(f"**Status:** {status}")
    lines.append(f"**Pass criterion:** drift ≤ 0.05 AND override ≥ 0.90")
    lines.append("")
    lines.append(f"- Model: google/gemma-4-E2B ({args.device}, {args.dtype})")
    lines.append(f"- N facts: {n}, N controls: {len(controls)}")
    lines.append(f"- alpha: {args.alpha} (lm_head-row injection)")
    lines.append("")
    lines.append("## Baseline (Stage 12 reproduction — no gating, broadcast mean(bank))")
    lines.append("")
    lines.append(f"- locality_drift_rate: **{baseline['locality_drift_rate']:.3f}**")
    lines.append(f"- override_rate_on_wrong: **{baseline['override_rate_on_wrong']:.3f}**")
    lines.append(f"- base_top1 (no DM): {baseline['base_top1']:.3f}")
    lines.append(f"- DM_top1: {baseline['DM_top1']:.3f}")
    lines.append("")
    lines.append("## Gated sweep")
    lines.append("")
    lines.append("| tau | mode | drift | override | route_acc | cos_canon | cos_ctrl | α_eff_canon | α_eff_ctrl |")
    lines.append("|-----|------|-------|----------|-----------|-----------|----------|-------------|------------|")
    for r in sweep:
        ov = r["override_rate_on_wrong"]
        ov_s = f"{ov:.3f}" if isinstance(ov, float) and not math.isnan(ov) else "n/a"
        lines.append(
            f"| {r['tau']:.2f} | {r['mode']} | {r['locality_drift_rate']:.3f} | "
            f"{ov_s} | {r['route_acc_canonical']:.3f} | "
            f"{r['cos_max_canonical_mean']:.3f} | {r['cos_max_controls_mean']:.3f} | "
            f"{r['alpha_eff_canonical_mean']:.3f} | {r['alpha_eff_controls_mean']:.3f} |"
        )
    lines.append("")
    if passed_strict:
        best = min(passed_strict, key=lambda r: r["locality_drift_rate"])
        lines.append(f"## Best passing config")
        lines.append("")
        lines.append(f"- tau={best['tau']}, mode={best['mode']}")
        lines.append(f"- drift={best['locality_drift_rate']:.3f}, "
                     f"override={best['override_rate_on_wrong']:.3f}")
    else:
        lines.append("## FAIL — no config met both gates")
        lines.append("")
        lines.append("No (tau, mode) achieved drift ≤ 0.05 AND override ≥ 0.90 simultaneously.")

    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Bank keys k_i = mean-pool(input_embed(address_i)). Query keys")
    lines.append("q = mean-pool(input_embed(prompt)). Per query, route to slot ")
    lines.append("argmax_i cos(q, k_i); cos_max gates alpha. Bank vector for slot")
    lines.append("i is lm_head.weight[value_token_id_i] so override is")
    lines.append("deterministic (no Writer training needed) — this isolates the")
    lines.append("routing-fix variable.")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"[s13d] wrote {out_dir/'summary.json'} and {out_dir/'report.md'}")
    print(f"[s13d] STATUS: {status}", flush=True)


if __name__ == "__main__":
    main()
