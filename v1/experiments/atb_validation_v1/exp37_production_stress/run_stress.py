"""Exp37 — Production Stress.

Three sub-experiments:
  37.A  Capability benchmark (lm-eval-harness lite — HellaSwag/ARC-easy)
  37.B  TOST falsification (equivalence test between routed and base on
        non-fact tasks: small mean diff with 90% CI inside [-0.02, 0.02])
  37.C  Forgetting probes — 50 known-good facts read post-patch; verify
        margin doesn't degrade > 0.5 nats relative to base.

All sub-experiments assume bank.pt and (for B/C) a trained router from
exp35b/run_qwen_exp35b/router_10k.pt.
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import json
import math
import random
import statistics
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


def load_setup(args):
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] bank {args.bank}", flush=True)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()
    return tok, model, entries, device, dtype, W_ref


def random_patch_set(entries, k, seed=0):
    rng = random.Random(seed)
    fids = rng.sample(list(entries.keys()), k)
    return fids


@torch.no_grad()
def hellaswag_score(model, tok, n=200, device="mps"):
    """Lite HellaSwag accuracy: pick continuation with lowest NLL."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(n))
    correct = 0
    total = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        ctx = ex["ctx"]
        ends = ex["endings"]
        label = int(ex["label"])
        scores = []
        for end in ends:
            full = ctx + " " + end
            enc = tok(full, return_tensors="pt", truncation=True, max_length=160).to(device)
            ctx_len = len(tok(ctx, return_tensors="pt").input_ids[0])
            ids = enc.input_ids
            if ids.size(1) - ctx_len < 1:
                scores.append(float("inf"))
                continue
            with torch.no_grad():
                out = model(**enc, use_cache=False)
            logits = out.logits[0, ctx_len-1:-1]
            tgt = ids[0, ctx_len:]
            nll = F.cross_entropy(logits.float(), tgt, reduction="sum").item()
            scores.append(nll / max(1, tgt.size(0)))
        pred = int(torch.tensor(scores).argmin().item())
        correct += int(pred == label)
        total += 1
        if (i + 1) % 50 == 0:
            print(f"  hellaswag {i+1}/{n} acc={correct/total:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return correct / max(1, total)


def run_37_A(args, tok, model, entries, device, dtype, W_ref):
    """Capability: HellaSwag on base vs patched (k=10, k=100)."""
    print("[37.A] base hellaswag", flush=True)
    base_acc = hellaswag_score(model, tok, n=args.n_hellaswag, device=device)

    results = {"base_acc": base_acc, "n": args.n_hellaswag}
    for k in [10, 100]:
        fids = random_patch_set(entries, k, seed=0)
        factors = [(entries[f]["b"].to(device, dtype=dtype),
                    entries[f]["a"].to(device, dtype=dtype)) for f in fids]
        W_old = apply_factors(model, args.edit_layer, factors)
        try:
            print(f"[37.A] patched k={k} hellaswag", flush=True)
            patched_acc = hellaswag_score(model, tok, n=args.n_hellaswag, device=device)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)
        drop = base_acc - patched_acc
        results[f"k{k}_acc"] = patched_acc
        results[f"k{k}_abs_drop"] = drop
        results[f"k{k}_rel_drop"] = drop / max(1e-9, base_acc)
    results["pre_registered_max_rel_drop_k10"] = 0.05
    results["pre_registered_max_rel_drop_k100"] = 0.10
    results["pass"] = (
        results.get("k10_rel_drop", 1.0) <= 0.05 and
        results.get("k100_rel_drop", 1.0) <= 0.10
    )
    return results


def tost_equivalence(diffs, eps=0.02, alpha=0.05):
    """Two One-Sided Tests for equivalence within [-eps, +eps]."""
    n = len(diffs)
    if n < 3:
        return {"pass": False, "n": n, "_msg": "too few"}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if se == 0:
        return {"mean": mean, "sd": sd, "n": n, "pass": abs(mean) < eps}
    # t-stat for H0_low: mean <= -eps  (reject if t > t_crit)
    t1 = (mean - (-eps)) / se
    t2 = ((eps) - mean) / se
    # use normal approx with n large
    z_crit = 1.645  # one-sided 0.05
    pass_ = (t1 > z_crit) and (t2 > z_crit)
    return {"mean_diff": mean, "sd": sd, "n": n, "se": se,
            "t_lower": t1, "t_upper": t2, "eps": eps, "pass": pass_}


@torch.no_grad()
def neutral_logp_diff(model, tok, sentences, device):
    """Average per-token logp on neutral sentences."""
    out = []
    for s in sentences:
        enc = tok(s, return_tensors="pt", truncation=True, max_length=64).to(device)
        if enc.input_ids.size(1) < 3: continue
        o = model(**enc, use_cache=False)
        logits = o.logits[0, :-1].float()
        tgt = enc.input_ids[0, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(1, tgt.unsqueeze(1)).squeeze().mean().item()
        out.append(lp)
    return out


def run_37_B(args, tok, model, entries, device, dtype, W_ref):
    """TOST: per-sentence base_logp vs patched_logp diff on neutral wikitext.
    Patch is k=10 random facts. Equivalence within ±0.02 nats per token."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    subjects_lower = set(e["subject"].lower() for e in entries.values())
    neutral = []
    for x in ds:
        t = x["text"].strip()
        if not t or len(t) < 40: continue
        tl = t.lower()
        if any(s in tl for s in subjects_lower if len(s) > 4): continue
        neutral.append(t)
        if len(neutral) >= args.n_tost: break

    print(f"[37.B] {len(neutral)} sentences, base pass", flush=True)
    base_lp = neutral_logp_diff(model, tok, neutral, device)

    fids = random_patch_set(entries, 10, seed=0)
    factors = [(entries[f]["b"].to(device, dtype=dtype),
                entries[f]["a"].to(device, dtype=dtype)) for f in fids]
    W_old = apply_factors(model, args.edit_layer, factors)
    try:
        print(f"[37.B] patched pass", flush=True)
        patched_lp = neutral_logp_diff(model, tok, neutral, device)
    finally:
        restore(model, args.edit_layer, W_old)
    assert_bit_equal(model, args.edit_layer, W_ref)

    diffs = [b - p for b, p in zip(base_lp, patched_lp)]
    tost = tost_equivalence(diffs, eps=args.tost_eps)
    tost["_caveat"] = "k=10 patch; equivalence in mean per-token logp"
    return tost


def run_37_C(args, tok, model, entries, device, dtype, W_ref):
    """Forgetting probes: pick 50 random facts as 'known-good' probes
    (target_true vs target_new margin on the BASE model). Then patch
    50 unrelated other facts (k=50), re-measure probe margins. Drop should be
    bounded."""
    rng = random.Random(1)
    all_ids = list(entries.keys())
    probes = rng.sample(all_ids, 50)
    others = [x for x in all_ids if x not in probes]
    patch_ids = rng.sample(others, 50)

    # load prompts from splits (bank entries don't carry prompt text)
    splits_dir = EXP35B / "data" / "splits"
    id2prompt = {}
    for s in ("train", "val", "test"):
        for r in json.load(open(splits_dir / f"{s}.json")):
            id2prompt[r["id"]] = r["prompt"].format(r["subject"])

    # base margins on probe canonical prompts
    base_margins = []
    for fid in probes:
        e = entries[fid]
        t_new = first_target_id(tok, e["target_new"])
        t_true = first_target_id(tok, e["target_true"])
        m = margin_at_last(model, tok, id2prompt[fid], t_new, t_true)
        base_margins.append(m)

    factors = [(entries[f]["b"].to(device, dtype=dtype),
                entries[f]["a"].to(device, dtype=dtype)) for f in patch_ids]
    W_old = apply_factors(model, args.edit_layer, factors)
    try:
        patched_margins = []
        for fid in probes:
            e = entries[fid]
            t_new = first_target_id(tok, e["target_new"])
            t_true = first_target_id(tok, e["target_true"])
            m = margin_at_last(model, tok, id2prompt[fid], t_new, t_true)
            patched_margins.append(m)
    finally:
        restore(model, args.edit_layer, W_old)
    assert_bit_equal(model, args.edit_layer, W_ref)

    drops = [b - p for b, p in zip(base_margins, patched_margins)]
    abs_drops = [abs(d) for d in drops]
    pass_frac = sum(1 for d in abs_drops if d < 0.5) / max(1, len(abs_drops))
    return {"n_probes": len(probes), "n_patched": len(patch_ids),
            "mean_abs_drop": sum(abs_drops)/max(1,len(abs_drops)),
            "frac_abs_drop_below_0p5": pass_frac,
            "pre_registered_min": 0.9, "pass": pass_frac >= 0.9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(EXP35B / "data" / "bank.pt"))
    ap.add_argument("--test", default="all", choices=["all", "37.A", "37.B", "37.C"])
    ap.add_argument("--n-hellaswag", type=int, default=300)
    ap.add_argument("--n-tost", type=int, default=500)
    ap.add_argument("--tost-eps", type=float, default=0.02)
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp37"))
    args = ap.parse_args()

    seed_everything(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tok, model, entries, device, dtype, W_ref = load_setup(args)
    runners = {"37.A": run_37_A, "37.B": run_37_B, "37.C": run_37_C}
    tests = list(runners.keys()) if args.test == "all" else [args.test]

    results = {}
    if (out / "results.json").exists():
        results = json.load(open(out / "results.json"))
    for t in tests:
        print(f"\n=== {t} ===", flush=True)
        results[t] = runners[t](args, tok, model, entries, device, dtype, W_ref)
        json.dump(results, open(out / "results.json", "w"), indent=2)
        print(json.dumps(results[t], indent=2), flush=True)

    overall_pass = all(r.get("pass", False) for r in results.values())
    results["_overall_pass"] = overall_pass
    json.dump(results, open(out / "results.json", "w"), indent=2)
    print(f"\n=== OVERALL PASS = {overall_pass} ===", flush=True)


if __name__ == "__main__":
    main()
