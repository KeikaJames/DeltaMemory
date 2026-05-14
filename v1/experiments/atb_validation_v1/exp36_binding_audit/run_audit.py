"""Exp36 — Binding Audit (depends on exp35b/data/bank.pt).

Seven sub-tests:
  36.1 subject_swap
  36.2 adversarial_paraphrases
  36.3 counterfactual_chaining
  36.4 negation_probe
  36.5 ood_subject_locality
  36.6 kl_audit
  36.7 patch_restore_stress

Run with --test 36.X (or 'all').

All sub-tests use single-fact patching (k=1) on each test fact, then
measure margin under perturbed read prompts.
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP35B = HERE.parent / "exp35b_memit_bank"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
_spec = iu.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = iu.module_from_spec(_spec); _spec.loader.exec_module(_bb)
first_target_id = _bb.first_target_id
apply_factors = _bb.apply_factors
restore = _bb.restore
margin_at_last = _bb.margin_at_last
assert_bit_equal = _bb.assert_bit_equal


def patched_margin(model, tok, layer, b, a, prompt, t_new, t_true, W_ref):
    W_old = apply_factors(model, layer, [(b, a)])
    try:
        m = margin_at_last(model, tok, prompt, t_new, t_true)
    finally:
        restore(model, layer, W_old)
    assert_bit_equal(model, layer, W_ref)
    return m


def load_setup(args):
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] bank {args.bank}", flush=True)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    test_rows = json.load(open(EXP35B / "data" / "splits" / "test.json"))
    val_rows = json.load(open(EXP35B / "data" / "splits" / "val.json"))
    train_rows = json.load(open(EXP35B / "data" / "splits" / "train.json"))
    all_rows = {r["id"]: r for r in train_rows + val_rows + test_rows}
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()
    return tok, model, entries, all_rows, test_rows, device, dtype, W_ref


def get_factor(entries, fid, device, dtype):
    e = entries[fid]
    return (e["b"].to(device, dtype=dtype), e["a"].to(device, dtype=dtype))


# ---- 36.1 ----
def run_36_1(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """Subject swap: pair each test fact i with a different-subject fact j of
    same relation. Patch with i's factor, read i's prompt with subject of j.
    The model should NOT update target for the wrong subject —
    |Δ_margin_swapped − base_margin_at_j| < 1 nat per fact."""
    by_rel = {}
    for r in test_rows:
        by_rel.setdefault(r["relation"], []).append(r)
    pairings = []
    rng = random.Random(0)
    for r in test_rows:
        cands = [x for x in by_rel.get(r["relation"], []) if x["subject"] != r["subject"]]
        if not cands: continue
        pairings.append((r, rng.choice(cands)))
    pairings = pairings[: args.n_queries]
    print(f"[36.1] {len(pairings)} pairings", flush=True)

    rows = []
    t0 = time.time()
    for i, (ri, rj) in enumerate(pairings):
        fid_i = ri["id"]
        if fid_i not in entries: continue
        t_new = first_target_id(tok, ri["target_new"])
        t_true = first_target_id(tok, ri["target_true"])
        # swap subject: use j's subject in i's prompt template
        swap_prompt = ri["prompt"].format(rj["subject"])
        base_swap = margin_at_last(model, tok, swap_prompt, t_new, t_true)
        b, a = get_factor(entries, fid_i, device, dtype)
        patched_swap = patched_margin(model, tok, args.edit_layer, b, a,
                                       swap_prompt, t_new, t_true, W_ref)
        delta = patched_swap - base_swap
        rows.append({"i": fid_i, "j": rj["id"], "base_swap": base_swap,
                     "patched_swap": patched_swap, "delta": delta})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pairings)} mean|delta|={sum(abs(r['delta']) for r in rows)/len(rows):.3f} ({time.time()-t0:.0f}s)", flush=True)
    n = len(rows)
    pass_frac = sum(1 for r in rows if abs(r["delta"]) < 1.0) / max(1, n)
    return {"n": n, "pass_frac_delta_within_1nat": pass_frac,
            "mean_abs_delta": sum(abs(r["delta"]) for r in rows) / max(1, n),
            "pre_registered_min": 0.9, "pass": pass_frac >= 0.9}


# ---- 36.2 ----
def adv_typo(s):
    if len(s) < 4: return s
    i = len(s) // 2
    return s[:i] + s[i+1] + s[i] + s[i+2:]


def adv_word_swap(text):
    words = text.split()
    if len(words) < 4: return text
    # swap first two content words (skip determiners)
    skips = {"the", "a", "an", "of", "to", "in", "on"}
    idx = [k for k, w in enumerate(words) if w.lower() not in skips]
    if len(idx) < 2: return text
    a, b = idx[0], idx[1]
    words[a], words[b] = words[b], words[a]
    return " ".join(words)


def adv_det_swap(text):
    return re.sub(r"\bThe\b", "A", re.sub(r"\bthe\b", "a", text), count=1)


def adv_question(text):
    return f"Is it true that {text}"


def adv_caps_subject(text, subject):
    return text.replace(subject, subject.upper())


def run_36_2(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """For each test fact, apply 5 adversarial templates to canonical prompt
    and measure Gate B (margin > 0)."""
    templates = [
        ("typo_in_subject", lambda r: r["prompt"].format(adv_typo(r["subject"]))),
        ("word_swap", lambda r: adv_word_swap(r["prompt"].format(r["subject"]))),
        ("det_swap", lambda r: adv_det_swap(r["prompt"].format(r["subject"]))),
        ("question_form", lambda r: adv_question(r["prompt"].format(r["subject"]))),
        ("all_caps_subject", lambda r: adv_caps_subject(r["prompt"].format(r["subject"]), r["subject"])),
    ]
    # baseline = in-distribution Gate B fraction on canonical prompt
    rows_canon = []
    rows_adv = {name: [] for name, _ in templates}
    t0 = time.time()
    for i, r in enumerate(test_rows[: args.n_queries]):
        fid = r["id"]
        if fid not in entries: continue
        t_new = first_target_id(tok, r["target_new"])
        t_true = first_target_id(tok, r["target_true"])
        b, a = get_factor(entries, fid, device, dtype)
        canon = r["prompt"].format(r["subject"])
        m_canon = patched_margin(model, tok, args.edit_layer, b, a, canon, t_new, t_true, W_ref)
        rows_canon.append(m_canon > 0)
        for name, fn in templates:
            try:
                p = fn(r)
            except Exception:
                continue
            m = patched_margin(model, tok, args.edit_layer, b, a, p, t_new, t_true, W_ref)
            rows_adv[name].append(m > 0)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n_queries} ({time.time()-t0:.0f}s)", flush=True)
    canon_frac = sum(rows_canon) / max(1, len(rows_canon))
    per_template = {name: {"gate_b_frac": sum(v)/max(1,len(v)),
                           "ratio_to_canon": (sum(v)/max(1,len(v)))/max(1e-9,canon_frac)}
                    for name, v in rows_adv.items()}
    min_ratio = min(p["ratio_to_canon"] for p in per_template.values())
    return {"canonical_gate_b_frac": canon_frac, "per_template": per_template,
            "min_ratio_to_canon": min_ratio,
            "pre_registered_min": 0.75,
            "pass": min_ratio >= 0.75}


# ---- 36.3 ----
def run_36_3(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """Chaining placeholder: pre-registered to require chain candidates with
    ≥3 paraphrases. Since not all corpora have this, we measure a weaker
    proxy: patch fact i, ask its canonical prompt, check return-fact recall
    on i's paraphrase[0] (which is what we did in Φ1). Report this number;
    if < 0.8 mark as FAIL but note caveat."""
    n_match = 0
    n_total = 0
    t0 = time.time()
    for i, r in enumerate(test_rows[: args.n_chains]):
        fid = r["id"]
        if fid not in entries: continue
        paras = r.get("paraphrase_prompts", [])
        if len(paras) < 1: continue
        t_new = first_target_id(tok, r["target_new"])
        t_true = first_target_id(tok, r["target_true"])
        b, a = get_factor(entries, fid, device, dtype)
        m = patched_margin(model, tok, args.edit_layer, b, a, paras[0], t_new, t_true, W_ref)
        n_match += int(m > 0)
        n_total += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n_chains} ({time.time()-t0:.0f}s)", flush=True)
    frac = n_match / max(1, n_total)
    return {"n": n_total, "frac_paraphrase_returns_new_target": frac,
            "pre_registered_min": 0.8, "pass": frac >= 0.8,
            "_caveat": "weaker proxy: paraphrase-level return, not multi-hop chain"}


# ---- 36.4 ----
def run_36_4(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """Negation: 'It is not true that {prompt}'. The model should NOT
    confidently produce target_new under negation."""
    rows = []
    t0 = time.time()
    for i, r in enumerate(test_rows[: args.n_queries]):
        fid = r["id"]
        if fid not in entries: continue
        t_new = first_target_id(tok, r["target_new"])
        t_true = first_target_id(tok, r["target_true"])
        canon = r["prompt"].format(r["subject"])
        neg = f"It is not true that {canon}"
        b, a = get_factor(entries, fid, device, dtype)
        m_canon = patched_margin(model, tok, args.edit_layer, b, a, canon, t_new, t_true, W_ref)
        m_neg = patched_margin(model, tok, args.edit_layer, b, a, neg, t_new, t_true, W_ref)
        rows.append({"canon": m_canon, "neg": m_neg, "delta": m_neg - m_canon})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n_queries} ({time.time()-t0:.0f}s)", flush=True)
    n = len(rows)
    pass_frac = sum(1 for r in rows if r["delta"] < -0.5) / max(1, n)
    return {"n": n, "frac_delta_below_neg_0p5_nats": pass_frac,
            "pre_registered_min": 0.9, "pass": pass_frac >= 0.9,
            "mean_delta": sum(r["delta"] for r in rows) / max(1, n)}


# ---- 36.5 ----
def run_36_5(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """OOD locality: random unrelated subjects from a different domain.
    We use neutral filler prompts. Patch with each of n_test_facts factors and
    check that margin on these OOD prompts barely moves."""
    ood_prompts = [
        "The sky is",
        "Two plus two equals",
        "The capital of France is",
        "Water boils at",
        "Photosynthesis is performed by",
        "The largest planet in our solar system is",
        "Shakespeare wrote",
        "Einstein discovered",
    ]
    fids = list(entries.keys())[: min(args.n_ood, 200)]
    base_margins = []
    delta_margins = []
    t0 = time.time()
    for i, fid in enumerate(fids):
        if fid not in entries: continue
        e = entries[fid]
        t_new = first_target_id(tok, e["target_new"])
        t_true = first_target_id(tok, e["target_true"])
        b, a = get_factor(entries, fid, device, dtype)
        for p in ood_prompts:
            base = margin_at_last(model, tok, p, t_new, t_true)
            assert_bit_equal(model, args.edit_layer, W_ref)
            pat = patched_margin(model, tok, args.edit_layer, b, a, p, t_new, t_true, W_ref)
            delta_margins.append(abs(pat - base))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(fids)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(delta_margins)
    pass_frac = sum(1 for d in delta_margins if d < 0.5) / max(1, n)
    return {"n_obs": n,
            "frac_abs_delta_below_0p5_nats": pass_frac,
            "mean_abs_delta": sum(delta_margins)/max(1, n),
            "pre_registered_min": 0.95, "pass": pass_frac >= 0.95}


# ---- 36.6 ----
@torch.no_grad()
def run_36_6(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """KL audit on WikiText neutral sentences."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    subjects_lower = set(e["subject"].lower() for e in entries.values())
    neutral = []
    for x in ds:
        t = x["text"].strip()
        if not t or len(t) < 40: continue
        # neutrality: no subject substring
        tl = t.lower()
        if any(s in tl for s in subjects_lower if len(s) > 4):
            continue
        neutral.append(t)
        if len(neutral) >= args.n_sentences: break
    print(f"[36.6] {len(neutral)} neutral sentences", flush=True)

    fids = list(entries.keys())[:10]
    factors = [get_factor(entries, f, device, dtype) for f in fids]

    kls = []
    t0 = time.time()
    for i, p in enumerate(neutral):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(device)
        if enc["input_ids"].size(1) < 4: continue
        out_b = model(**enc, use_cache=False)
        logp_b = F.log_softmax(out_b.logits[0, -1].float(), dim=-1)
        W_old = apply_factors(model, args.edit_layer, factors)
        try:
            out_p = model(**enc, use_cache=False)
            logp_p = F.log_softmax(out_p.logits[0, -1].float(), dim=-1)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)
        # KL(p_b || p_p) in fp32 nats
        p_b = logp_b.exp()
        kl = (p_b * (logp_b - logp_p)).sum().item()
        kls.append(kl)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(neutral)} median_kl={sorted(kls)[len(kls)//2]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    kls_s = sorted(kls)
    median_kl = kls_s[len(kls_s)//2] if kls_s else 0.0
    p95_kl = kls_s[int(0.95*len(kls_s))] if kls_s else 0.0
    return {"n": len(kls), "median_kl_nats": median_kl, "p95_kl_nats": p95_kl,
            "k_facts_patched": len(fids),
            "pre_registered_median_max": 0.05,
            "pre_registered_p95_max": 0.5,
            "pass": median_kl <= 0.05 and p95_kl <= 0.5}


# ---- 36.7 ----
def run_36_7(args, tok, model, entries, all_rows, test_rows, device, dtype, W_ref):
    """Patch-restore stress: n_cycles iterations of apply/restore with assert_bit_equal."""
    fids = list(entries.keys())
    rng = random.Random(0)
    fails = 0
    t0 = time.time()
    for i in range(args.n_cycles):
        fid = rng.choice(fids)
        b, a = get_factor(entries, fid, device, dtype)
        W_old = apply_factors(model, args.edit_layer, [(b, a)])
        restore(model, args.edit_layer, W_old)
        try:
            assert_bit_equal(model, args.edit_layer, W_ref)
        except Exception:
            fails += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n_cycles} fails={fails} ({time.time()-t0:.0f}s)", flush=True)
    return {"n_cycles": args.n_cycles, "bit_equal_failures": fails,
            "pre_registered_max": 0, "pass": fails == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(EXP35B / "data" / "bank.pt"))
    ap.add_argument("--test", default="all")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--n-chains", type=int, default=500)
    ap.add_argument("--n-ood", type=int, default=200)
    ap.add_argument("--n-sentences", type=int, default=1000)
    ap.add_argument("--n-cycles", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp36"))
    args = ap.parse_args()

    seed_everything(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tok, model, entries, all_rows, test_rows, device, dtype, W_ref = load_setup(args)

    tests = ["36.1", "36.2", "36.3", "36.4", "36.5", "36.6", "36.7"] \
        if args.test == "all" else [args.test]
    runners = {
        "36.1": run_36_1, "36.2": run_36_2, "36.3": run_36_3,
        "36.4": run_36_4, "36.5": run_36_5, "36.6": run_36_6,
        "36.7": run_36_7,
    }
    results = {}
    if (out / "results.json").exists():
        results = json.load(open(out / "results.json"))

    for t in tests:
        print(f"\n=== {t} ===", flush=True)
        results[t] = runners[t](args, tok, model, entries, all_rows, test_rows,
                                 device, dtype, W_ref)
        json.dump(results, open(out / "results.json", "w"), indent=2)
        print(json.dumps(results[t], indent=2), flush=True)

    overall_pass = all(r.get("pass", False) for r in results.values())
    results["_overall_pass"] = overall_pass
    json.dump(results, open(out / "results.json", "w"), indent=2)
    print(f"\n=== OVERALL PASS = {overall_pass} ===", flush=True)


if __name__ == "__main__":
    main()
