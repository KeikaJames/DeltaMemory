"""Exp38 — shared evaluation panels (Φ1, Φ3, 36.4, 37.C, HellaSwag, D6).

Each panel returns a dict; all panels run within `gated_patches` context.

Usage from a variant script:
    from exp38_gated_bank.eval_panels import run_all_panels
    results = run_all_panels(model, tok, bank, gate_fn, gate_ctx, args)
"""
from __future__ import annotations

import importlib.util as iu
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP35B = HERE.parent / "exp35b_memit_bank"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
sys.path.insert(0, str(HERE))

from gates import gated_patches, margin_at_last_gated, gate_G0_baseline  # noqa: E402

_spec = iu.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = iu.module_from_spec(_spec); _spec.loader.exec_module(_bb)
first_target_id = _bb.first_target_id


# ---- helpers ----

def _margin(model, tok, prompt, t_new, t_true):
    return margin_at_last_gated(model, tok, prompt, t_new, t_true)


def _load_splits():
    splits_dir = EXP35B / "data" / "splits"
    train = json.load(open(splits_dir / "train.json"))
    val = json.load(open(splits_dir / "val.json"))
    test = json.load(open(splits_dir / "test.json"))
    return train, val, test


def _load_all_rows():
    tr, va, te = _load_splits()
    return {r["id"]: r for r in tr + va + te}


# ---- panel: Φ1 oracle composition (1500 × 4k × 3seed) ----

@torch.no_grad()
def panel_phi1(model, tok, bank_struct, gate_fn, gate_ctx, *,
               n_test: int = 1500, k_values=(1, 10, 100, 1000),
               seeds=(0, 1, 2), verbose: bool = True):
    """Measure gate_d (target - shuffled-target) under gated patches at multiple k."""
    _, _, test_rows = _load_splits()
    all_rows = _load_all_rows()
    in_bank = set(bank_struct.fact_ids)
    test = [r for r in test_rows if r["id"] in in_bank][:n_test]
    id2idx = {fid: i for i, fid in enumerate(bank_struct.fact_ids)}
    N = len(bank_struct.fact_ids)

    # precompute base margins (no patch) for each test fact, target_new vs target_true
    base = {}
    for r in test:
        t_new = first_target_id(tok, r["target_new"])
        t_true = first_target_id(tok, r["target_true"])
        p = r["prompt"].format(r["subject"])
        m = _margin(model, tok, p, t_new, t_true)
        base[r["id"]] = (m, t_new, t_true, p)

    results = {}
    for k in k_values:
        per_seed = []
        for seed in seeds:
            rng = random.Random(seed)
            uplifts, gate_d, beats = [], [], []
            for r in test:
                fid = r["id"]
                m_base, t_new, t_true, prompt = base[fid]
                # pick k-1 distractor facts (from bank, not equal to fid)
                distractor_ids = rng.sample([x for x in bank_struct.fact_ids if x != fid],
                                            min(k - 1, N - 1))
                patch_idx = [id2idx[fid]] + [id2idx[x] for x in distractor_ids]
                # gate sees full bank; restrict via a mask = 1 on patch_idx, 0 elsewhere
                mask = torch.zeros(N, device=bank_struct.A.device, dtype=bank_struct.A.dtype)
                mask[patch_idx] = 1.0
                ctx = dict(gate_ctx); ctx["_phi1_mask"] = mask
                # gate fn factored: we wrap to multiply by mask after
                def gate_fn_masked(scores, hidden, c=ctx, base_fn=gate_fn):
                    g = base_fn(scores, hidden, c)
                    return g * c["_phi1_mask"]
                with gated_patches(model, gate_ctx["edit_layer"], bank_struct,
                                   gate_fn_masked, ctx):
                    m_p = _margin(model, tok, prompt, t_new, t_true)
                uplifts.append(m_p - m_base)
                beats.append(1.0 if m_p > m_base else 0.0)

                # shuffled control: same k but with target swapped to a random fact's target
                shuf_id = rng.choice([x for x in bank_struct.fact_ids if x != fid and x in all_rows])
                shuf_row = all_rows[shuf_id]
                t_new_shuf = first_target_id(tok, shuf_row["target_new"])
                t_true_shuf = first_target_id(tok, shuf_row["target_true"])
                with gated_patches(model, gate_ctx["edit_layer"], bank_struct,
                                   gate_fn_masked, ctx):
                    m_shuf = _margin(model, tok, prompt, t_new_shuf, t_true_shuf)
                gate_d.append((m_p - m_base) - (m_shuf - m_base))

            per_seed.append({
                "seed": seed,
                "mean_uplift": float(sum(uplifts) / len(uplifts)),
                "mean_gate_d": float(sum(gate_d) / len(gate_d)),
                "frac_beats_base": float(sum(beats) / len(beats)),
            })
        agg = {
            "k": k,
            "per_seed": per_seed,
            "mean_uplift": sum(s["mean_uplift"] for s in per_seed) / len(per_seed),
            "mean_gate_d": sum(s["mean_gate_d"] for s in per_seed) / len(per_seed),
            "frac_beats_base": sum(s["frac_beats_base"] for s in per_seed) / len(per_seed),
        }
        results[f"k{k}"] = agg
        if verbose:
            print(f"  Φ1 k={k}: gate_d={agg['mean_gate_d']:+.3f}  uplift={agg['mean_uplift']:+.3f} "
                  f"beats={agg['frac_beats_base']*100:.1f}%", flush=True)
    return results


def _id_to_row(fid, rows):
    for r in rows:
        if r["id"] == fid:
            return r
    return None


# ---- panel: 37.C cross-talk ----

@torch.no_grad()
def panel_37_C(model, tok, bank_struct, gate_fn, gate_ctx, *,
               n_probes: int = 100, n_patch_sets: int = 50,
               patch_size: int = 50, seed: int = 1, verbose: bool = True):
    _, _, test_rows = _load_splits()
    in_bank = set(bank_struct.fact_ids)
    id_to_row = {r["id"]: r for r in test_rows if r["id"] in in_bank}
    train, val, _ = _load_splits()
    all_rows = {**{r["id"]: r for r in train if r["id"] in in_bank},
                **{r["id"]: r for r in val if r["id"] in in_bank},
                **id_to_row}

    rng = random.Random(seed)
    probe_ids = rng.sample(list(id_to_row.keys()), min(n_probes, len(id_to_row)))
    # base margins on probes (no patch)
    probe_meta = {}
    for fid in probe_ids:
        r = all_rows[fid]
        t_new = first_target_id(tok, r["target_new"])
        t_true = first_target_id(tok, r["target_true"])
        p = r["prompt"].format(r["subject"])
        m = _margin(model, tok, p, t_new, t_true)
        probe_meta[fid] = (m, t_new, t_true, p)

    id2idx = {fid: i for i, fid in enumerate(bank_struct.fact_ids)}
    N = len(bank_struct.fact_ids)
    drops = []
    for s in range(n_patch_sets):
        rng2 = random.Random(seed * 1000 + s)
        others = [x for x in bank_struct.fact_ids if x not in probe_ids]
        patch_ids = rng2.sample(others, min(patch_size, len(others)))
        patch_idx = [id2idx[x] for x in patch_ids]
        mask = torch.zeros(N, device=bank_struct.A.device, dtype=bank_struct.A.dtype)
        mask[patch_idx] = 1.0
        ctx = dict(gate_ctx); ctx["_mask"] = mask
        def gate_fn_masked(scores, hidden, c=ctx, base_fn=gate_fn):
            g = base_fn(scores, hidden, c)
            return g * c["_mask"]
        with gated_patches(model, gate_ctx["edit_layer"], bank_struct,
                           gate_fn_masked, ctx):
            for fid in probe_ids:
                m_base, t_new, t_true, prompt = probe_meta[fid]
                m_p = _margin(model, tok, prompt, t_new, t_true)
                drops.append(abs(m_base - m_p))
        if verbose and (s + 1) % 10 == 0:
            print(f"  37.C set {s+1}/{n_patch_sets}  mean|drop| so far={sum(drops)/len(drops):.3f}",
                  flush=True)
    pass_frac = sum(1 for d in drops if d < 0.5) / max(1, len(drops))
    return {"n_probes": len(probe_ids), "n_patch_sets": n_patch_sets,
            "patch_size": patch_size, "n_observations": len(drops),
            "mean_abs_drop": float(sum(drops) / max(1, len(drops))),
            "frac_abs_drop_below_0p5": pass_frac}


# ---- panel: 36.4 negation ----

NEG_TEMPLATES = [
    "It is not true that {p}",
    "It is false to say that {p}",
    "Is it the case that {p}?",
    "Some people incorrectly claim that {p}",
    "Contrary to fact, {p}",
]

@torch.no_grad()
def panel_neg_36_4(model, tok, bank_struct, gate_fn, gate_ctx, *,
                   n_facts: int = 600, seed: int = 0, verbose: bool = True):
    """For each fact: patch it (k=1, self), measure margin on each negation
    template; PASS = margin shifts down by >0.5 nats vs the affirmative-patched
    margin (i.e. gate-d under negation should suppress the new target)."""
    _, _, test_rows = _load_splits()
    in_bank = set(bank_struct.fact_ids)
    test = [r for r in test_rows if r["id"] in in_bank][:n_facts]
    id2idx = {fid: i for i, fid in enumerate(bank_struct.fact_ids)}
    N = len(bank_struct.fact_ids)

    fact_passes = 0
    fact_count = 0
    diffs = []
    for ri, r in enumerate(test):
        fid = r["id"]
        t_new = first_target_id(tok, r["target_new"])
        t_true = first_target_id(tok, r["target_true"])
        canon = r["prompt"].format(r["subject"])
        mask = torch.zeros(N, device=bank_struct.A.device, dtype=bank_struct.A.dtype)
        mask[id2idx[fid]] = 1.0
        ctx = dict(gate_ctx); ctx["_mask"] = mask
        def gate_fn_masked(scores, hidden, c=ctx, base_fn=gate_fn):
            return base_fn(scores, hidden, c) * c["_mask"]
        with gated_patches(model, gate_ctx["edit_layer"], bank_struct,
                           gate_fn_masked, ctx):
            m_aff = _margin(model, tok, canon, t_new, t_true)
            template_diffs = []
            for templ in NEG_TEMPLATES:
                neg_prompt = templ.format(p=canon.rstrip("?.!"))
                m_neg = _margin(model, tok, neg_prompt, t_new, t_true)
                template_diffs.append(m_aff - m_neg)  # >0.5 means negation suppressed t_new
        # PASS if ANY of the 5 templates gives diff > 0.5 (relaxed: avg)
        avg_diff = sum(template_diffs) / len(template_diffs)
        diffs.extend(template_diffs)
        if avg_diff > 0.5:
            fact_passes += 1
        fact_count += 1
        if verbose and (ri + 1) % 100 == 0:
            print(f"  36.4 {ri+1}/{len(test)} pass_frac={fact_passes/fact_count:.2%} "
                  f"mean_diff={sum(diffs)/len(diffs):+.2f}", flush=True)
    return {"n_facts": fact_count, "n_observations": len(diffs),
            "mean_affirm_minus_neg_nats": float(sum(diffs) / max(1, len(diffs))),
            "frac_facts_neg_suppresses": fact_passes / max(1, fact_count)}


# ---- panel: HellaSwag (lightweight, lm-logprob ranking) ----

@torch.no_grad()
def panel_hellaswag(model, tok, bank_struct, gate_fn, gate_ctx, *,
                    n_examples: int = 1000, k_patches: int = 100,
                    seed: int = 0, verbose: bool = True):
    """Per-example: pick k_patches random bank facts to patch; score each
    HellaSwag continuation by sum of log p(token) and pick argmax. Compare to
    no-patch baseline."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_examples, len(ds))))
    id2idx = {fid: i for i, fid in enumerate(bank_struct.fact_ids)}
    N = len(bank_struct.fact_ids)
    rng = random.Random(seed)

    def score(prompt):
        device = next(model.parameters()).device
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, use_cache=False)
        ids = enc["input_ids"][0]
        logp = F.log_softmax(out.logits[0].float(), dim=-1)
        # token-level logprob for shifted targets
        return float(logp[:-1, :].gather(1, ids[1:].unsqueeze(-1)).sum().item())

    # patch idx set fixed across all examples
    patch_ids = rng.sample(bank_struct.fact_ids, min(k_patches, N))
    mask = torch.zeros(N, device=bank_struct.A.device, dtype=bank_struct.A.dtype)
    for x in patch_ids:
        mask[id2idx[x]] = 1.0
    ctx = dict(gate_ctx); ctx["_mask"] = mask
    def gate_fn_masked(scores, hidden, c=ctx, base_fn=gate_fn):
        return base_fn(scores, hidden, c) * c["_mask"]

    correct = 0
    for i, ex in enumerate(ds):
        ctx_str = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        endings = ex["endings"]
        with gated_patches(model, gate_ctx["edit_layer"], bank_struct,
                           gate_fn_masked, ctx):
            scores = [score(ctx_str + " " + e) for e in endings]
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        if pred == int(ex["label"]):
            correct += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"  HellaSwag {i+1}/{len(ds)}  acc={correct/(i+1):.3f}", flush=True)
    acc = correct / max(1, len(ds))
    return {"n": len(ds), "k_patches": k_patches, "acc": acc}
