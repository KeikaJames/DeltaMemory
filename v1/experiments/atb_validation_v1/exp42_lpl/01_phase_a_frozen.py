"""Phase A — frozen-heads POC for LPL.

Goal: with no training, just force pauses at last_token across a few layers
in round 1, run K=2, and measure whether the bank-augmented round 2 logits
are MEASURABLY DIFFERENT (sanity) and whether NLL on a curated continuation
set moves favorably vs. base.

We use three small evaluation sets (no training data, all held-out):
  - simple_qa: classic 1-hop facts ("The capital of France is _Paris_").
  - negation: simple negated facts ("Paris is NOT in _Germany_").
  - 2hop: small 2-hop ("Marie Curie was born in Poland. The capital of
          her birth country is _Warsaw_").

We compare three settings:
  base          — unpatched model
  lpl_K1        — patched, force-pause OFF, K=1 (must match base, Gate 0)
  lpl_K2_pause  — patched, force pause at last_token in layers
                  [L//4, L//2, 3*L//4], K=2

Metric: mean NLL on the gold continuation tokens.

Decision: report; if `lpl_K2_pause < base` mean-NLL on ≥ 2 of 3 sets by any
margin, declare Phase A passes "movement direction" (we don't require +1pp
for frozen — that's Phase B with training).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn.functional as F

from atb_validation_v1._lib import load_model
from exp42_lpl import (
    AttentionBank,
    LPLHeads,
    LPLConfig,
    LPLRuntime,
    install_lpl_patch,
)


# -----------------------------------------------------------------------------
# Curated tiny eval set. Each entry: (prompt, gold_answer_string)
# All facts are extremely well-known so the base model should already know them
# — we're measuring whether the LPL-perturbation HURTS less / HELPS at all.

SIMPLE_QA = [
    ("Q: What is the capital of France?\nA:", " Paris"),
    ("Q: What is the capital of Japan?\nA:", " Tokyo"),
    ("Q: Who wrote Hamlet?\nA:", " William Shakespeare"),
    ("Q: What planet is known as the Red Planet?\nA:", " Mars"),
    ("Q: What is the largest ocean on Earth?\nA:", " The Pacific"),
    ("Q: What is the chemical symbol for gold?\nA:", " Au"),
    ("Q: In what year did World War II end?\nA:", " 1945"),
    ("Q: What is the speed of light in vacuum, in m/s? (just the number)\nA:", " 299"),
]

NEGATION = [
    ("Q: Is Paris in Germany? Answer yes or no.\nA:", " No"),
    ("Q: Is the Sun a planet? Answer yes or no.\nA:", " No"),
    ("Q: Is water made of carbon? Answer yes or no.\nA:", " No"),
    ("Q: Does ice float in water? Answer yes or no.\nA:", " Yes"),
    ("Q: Is the Pacific the smallest ocean? Answer yes or no.\nA:", " No"),
    ("Q: Did Shakespeare write Hamlet? Answer yes or no.\nA:", " Yes"),
]

TWO_HOP = [
    ("Marie Curie was born in Poland. The capital of her birth country is",
     " Warsaw"),
    ("Mount Fuji is in Japan. The capital of that country is",
     " Tokyo"),
    ("The Eiffel Tower is in France. The capital of that country is",
     " Paris"),
    ("Albert Einstein was born in Germany. The capital of his birth country is",
     " Berlin"),
    ("The Amazon River is mostly in Brazil. The capital of that country is",
     " Brasília"),
]

SETS = {"simple_qa": SIMPLE_QA, "negation": NEGATION, "two_hop": TWO_HOP}


# -----------------------------------------------------------------------------

def nll_on_answer(logits: torch.Tensor, input_ids: torch.Tensor,
                  ans_token_start: int) -> float:
    """Mean NLL over answer tokens for a single-batch sequence.

    logits: [1, T, V]
    input_ids: [1, T]
    ans_token_start: index of first answer token (≥1)
    """
    # Standard causal shift: logits at position t predict token at position t+1.
    # So to score input_ids[t] (for t >= ans_token_start), use logits[:, t-1, :].
    # ans tokens span positions [ans_token_start, T-1].
    if ans_token_start >= input_ids.shape[1]:
        return float("nan")
    pred_logits = logits[0, ans_token_start - 1 : -1, :]   # [n_ans, V]
    gold = input_ids[0, ans_token_start:]                  # [n_ans]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    nll = -log_probs.gather(1, gold.unsqueeze(-1)).squeeze(-1).mean().item()
    return nll


# -----------------------------------------------------------------------------

def make_force_mask(active_layers: set[int], prompt_len: int,
                    seq_len: int, device, *, pause_pos: int = -1):
    """Return a per-layer callable producing a [B,T] bool pause mask.

    Only pause for layers in active_layers, only at position pause_pos
    (default last), only at prompt tail (not during answer continuation).
    For our prompt-only forward, pause_pos = prompt_len - 1.
    """
    pause_idx = (seq_len + pause_pos) if pause_pos < 0 else pause_pos

    def fpm(layer_idx: int, round_idx: int, h_in):
        B, T, _ = h_in.shape
        if layer_idx not in active_layers:
            return None
        m = torch.zeros(B, T, dtype=torch.bool, device=h_in.device)
        if 0 <= pause_idx < T:
            m[:, pause_idx] = True
        return m
    return fpm


# -----------------------------------------------------------------------------

def eval_set(name, items, tok, model, runtime_base, runtime_lpl_k1,
             runtime_lpl_pause, active_layers, device):
    rows = []
    for prompt, gold in items:
        full = prompt + gold
        enc = tok(full, return_tensors="pt").to(device)
        # answer token boundary
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        ans_start = prompt_ids.shape[1]

        # base
        with torch.no_grad():
            base_logits = model(**enc, use_cache=False).logits
        nll_base = nll_on_answer(base_logits, enc.input_ids, ans_start)

        # lpl K=1 (must match base — sanity)
        with torch.no_grad():
            r1 = runtime_lpl_k1.forward(enc.input_ids, attention_mask=enc.attention_mask)
        nll_lpl_k1 = nll_on_answer(r1.logits, enc.input_ids, ans_start)

        # lpl K=2 pause: rebuild runtime per-example so seq_len matches
        seq_len = enc.input_ids.shape[1]
        fpm = make_force_mask(active_layers, prompt_len=ans_start,
                              seq_len=seq_len, device=device,
                              pause_pos=ans_start - 1)
        bank = runtime_lpl_pause.bank
        bank.clear()
        cfg2 = LPLConfig(K_max=2, enabled=True, force_pause_mask=fpm)
        from exp42_lpl.runtime import LPLRuntime as _RT
        rt2 = _RT(model, runtime_lpl_pause.heads, bank, cfg2)
        with torch.no_grad():
            r2 = rt2.forward(enc.input_ids, attention_mask=enc.attention_mask,
                             clear_bank=False)
        nll_pause = nll_on_answer(r2.logits, enc.input_ids, ans_start)

        rows.append({
            "prompt": prompt[:60],
            "gold": gold,
            "nll_base": nll_base,
            "nll_lpl_k1": nll_lpl_k1,
            "nll_lpl_pause_k2": nll_pause,
            "bank_size_after": r2.bank_total_size_after,
        })
    return rows


# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--out", default=str(HERE / "phase_a_results.json"))
    args = p.parse_args()

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    bank = AttentionBank(num_layers=n_layers, hidden_size=d,
                         device=args.device, dtype=torch.bfloat16,
                         max_per_layer=512)
    heads = LPLHeads.fresh(n_layers, d,
                           pause_bias=-20.0, bank_gate_bias=10.0, halt_bias=10.0,
                           device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    rt_base = None  # we just call model directly
    cfg_k1 = LPLConfig(K_max=1, enabled=True)
    rt_k1 = LPLRuntime(model, heads, bank, cfg_k1)

    # pause active layers: every L//4 — [L//4, L//2, 3L//4]
    active = {n_layers // 4, n_layers // 2, (3 * n_layers) // 4}
    print(f"[phase_a] n_layers={n_layers} pause at layers={sorted(active)}")

    # dummy runtime carrying heads + bank for re-use of bank ref
    rt_pause = LPLRuntime(model, heads, bank, LPLConfig(K_max=2, enabled=True))

    all_results = {}
    summary = {}
    for name, items in SETS.items():
        print(f"\n[phase_a] === {name}  ({len(items)} items) ===")
        rows = eval_set(name, items, tok, model, rt_base, rt_k1, rt_pause,
                        active, args.device)
        m_base = sum(r["nll_base"] for r in rows) / len(rows)
        m_k1   = sum(r["nll_lpl_k1"] for r in rows) / len(rows)
        m_p    = sum(r["nll_lpl_pause_k2"] for r in rows) / len(rows)
        max_diff_k1 = max(abs(r["nll_base"] - r["nll_lpl_k1"]) for r in rows)
        delta = m_p - m_base
        print(f"  mean NLL  base={m_base:.4f}  lpl_K1={m_k1:.4f}"
              f" (max|Δ|={max_diff_k1:.1e})  lpl_pause_K2={m_p:.4f}"
              f"  Δ_vs_base={delta:+.4f}")
        all_results[name] = rows
        summary[name] = {
            "n": len(items),
            "mean_nll_base": m_base,
            "mean_nll_lpl_k1": m_k1,
            "mean_nll_lpl_pause_k2": m_p,
            "delta_vs_base": delta,
            "max_diff_k1_vs_base": max_diff_k1,
        }

    # Decision: "movement-direction PASS" if pause < base on ≥ 2 sets.
    n_better = sum(1 for s in summary.values() if s["delta_vs_base"] < 0)
    verdict = "MOVEMENT_PASS" if n_better >= 2 else ("MIXED" if n_better == 1 else "FAIL")
    print(f"\n[phase_a] sets_better_than_base = {n_better}/3 → {verdict}")
    out = {"summary": summary, "rows": all_results,
           "active_pause_layers": sorted(active),
           "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[phase_a] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
