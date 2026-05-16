"""e19 — seed replication for Phase B2 canonical results.

Replicates the canonical Phase B2 training (Δ=−3.90 at seed 0, layer 9) across
seeds {0,1,2,3,4} and layers {9,21} to confirm the NLL improvement is stable
and reproducible.

Goal: Validate that Δ is not a single-seed accident. Also replicate layer-21
which showed Δ=−6.29 at seed 0 (better than layer 9).

Each (seed, layer) combination = one fresh model load, fresh seed-controlled
data split, fresh projector init, fresh training run.

Output:
  - cells/L{layer}_s{seed}.json  (10 cells)
  - e19_summary.json             (aggregated statistics)

Pass criteria:
  - Layer 9:  all 5 seeds Δ ≤ -2.0, std ≤ 1.5
  - Layer 21: all 5 seeds Δ ≤ -3.0, std ≤ 1.5

Runtime: ~3 min per cell × 10 cells = ~30 min on MPS.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False):
    """Two-round LPL forward: preload (round 1), read (round 2)."""
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    """Cross-entropy on answer tokens."""
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def build_canonical_split(entries, *, n_train, n_test, n_preload, seed):
    """Canonical split: use bank.pt's 'split' field, shuffle with seed."""
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    
    rng = random.Random(seed)
    rng.shuffle(train_keys)
    rng.shuffle(test_keys)
    
    train_keys = train_keys[:n_train]
    test_keys = test_keys[:n_test]
    
    # preload from train, disjoint from train_keys
    preload_pool = [k for k in train_keys if k not in set(train_keys)]
    if len(preload_pool) < n_preload:
        # fallback: take from full train set
        all_train = data_io.filter_keys(entries, split="train", solo_pass=True)
        preload_pool = [k for k in all_train if k not in set(train_keys)]
    
    rng.shuffle(preload_pool)
    preload_keys = preload_pool[:n_preload]
    
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    
    return train_items, test_items, preload_keys


def run_canonical_cell(*, seed, bank_layer, device, model_name, bank_pt,
                       n_train, n_eval, n_preload, steps, lr, rank):
    """Run one canonical training cell and return result dict."""
    print(f"\n{'='*60}")
    print(f"[e19] CELL: seed={seed}, layer={bank_layer}")
    print(f"{'='*60}")
    
    # set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    blob = data_io.load_bank_blob(bank_pt)
    entries = blob["entries"]
    
    train_items, test_items, preload_keys = build_canonical_split(
        entries, n_train=n_train, n_test=n_eval, n_preload=n_preload, seed=seed
    )
    print(f"[e19] train={len(train_items)} test={len(test_items)} preload={len(preload_keys)}")
    
    # load model
    tok, model = load_model(model_name, device=device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    
    # init bank + heads
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=device,
                         dtype=torch.bfloat16, max_per_layer=n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=device, dtype=torch.float32)
    install_lpl_patch(model)
    
    # preload b-vectors with seed-controlled shuffle
    b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                      device=device, dtype=torch.float32)
    
    # init projector with seed-controlled randomness
    P = make_projector(d, rank=rank).to(device).float()
    
    def apply_proj():
        with torch.no_grad():
            proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[bank_layer] = proj
        bank.tags[bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True
    
    apply_proj()
    
    # eval functions
    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    def eval_lpl(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
            apply_proj()
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    # before eval
    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    print(f"[e19] BEFORE: base={base:.4f}  real={pre_real:.4f}")
    
    # training
    rng = random.Random(seed)
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    t0 = time.time()
    
    for step in range(steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        
        # rebuild bank with grad-tracking projector
        bank.frozen = False
        proj = (b_raw + P(b_raw)).to(dtype=torch.bfloat16)
        bank.slots[bank_layer] = proj
        bank.tags[bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True
        
        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        
        if (step + 1) % 50 == 0 or step == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            print(f"  [e19] step {step+1}/{steps} loss(avg50)={recent:.4f} ({time.time()-t0:.1f}s)")
    
    print(f"[e19] training done in {time.time()-t0:.1f}s")
    
    # after eval
    post_real = eval_lpl(test_items)
    print(f"[e19] AFTER:  base={base:.4f}  real={post_real:.4f}")
    
    delta_real = base - post_real
    print(f"[e19] Δ_real = {delta_real:.4f}")
    
    result = {
        "seed": seed,
        "bank_layer": bank_layer,
        "model": model_name,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "rank": rank,
        "lr": lr,
        "steps": steps,
        "before": {"base": base, "real": pre_real},
        "after": {"base": base, "real": post_real},
        "delta_real": delta_real,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "runtime_sec": time.time() - t0,
    }
    
    return result


def aggregate_summary(cells_dir, layers, seeds):
    """Aggregate results from all cells and compute statistics."""
    results_by_layer = {layer: [] for layer in layers}
    
    # load all cell results
    for layer in layers:
        for seed in seeds:
            cell_path = cells_dir / f"L{layer}_s{seed}.json"
            if not cell_path.exists():
                print(f"[e19] WARNING: missing cell {cell_path}")
                continue
            data = json.loads(cell_path.read_text())
            results_by_layer[layer].append(data)
    
    # compute statistics per layer
    summary = {"layers": {}, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    
    for layer in layers:
        cell_results = results_by_layer[layer]
        if not cell_results:
            summary["layers"][str(layer)] = {"error": "no cells found"}
            continue
        
        deltas = [r["delta_real"] for r in cell_results]
        n = len(deltas)
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas, ddof=1) if n > 1 else 0.0
        
        # 95% CI using t-distribution
        from scipy import stats
        if n > 1:
            ci_margin = stats.t.ppf(0.975, n-1) * std_delta / np.sqrt(n)
        else:
            ci_margin = 0.0
        
        ci_lower = mean_delta - ci_margin
        ci_upper = mean_delta + ci_margin
        
        # pass criteria
        if layer == 9:
            thresh = -2.0
            std_thresh = 1.5
        elif layer == 21:
            thresh = -3.0
            std_thresh = 1.5
        else:
            thresh = -1.0
            std_thresh = 2.0
        
        all_beat_thresh = all(d <= thresh for d in deltas)
        std_pass = std_delta <= std_thresh
        
        summary["layers"][str(layer)] = {
            "n_seeds": n,
            "deltas": deltas,
            "mean_delta": float(mean_delta),
            "std_delta": float(std_delta),
            "ci_95": [float(ci_lower), float(ci_upper)],
            "all_seeds_beat_thresh": all_beat_thresh,
            "thresh": thresh,
            "std_pass": std_pass,
            "std_thresh": std_thresh,
            "pass": all_beat_thresh and std_pass,
        }
        
        print(f"\n[e19] Layer {layer} summary:")
        print(f"  Mean Δ = {mean_delta:.4f} ± {std_delta:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  All seeds ≤ {thresh}: {all_beat_thresh}")
        print(f"  Std ≤ {std_thresh}: {std_pass}")
        print(f"  PASS: {all_beat_thresh and std_pass}")
    
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed list")
    p.add_argument("--layers", default="9,21", help="Comma-separated layer list")
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--skip_existing", type=bool, default=True)
    args = p.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    layers = [int(l.strip()) for l in args.layers.split(",")]
    
    cells_dir = HERE / "cells"
    cells_dir.mkdir(exist_ok=True)
    
    print(f"[e19] Seed replication driver")
    print(f"[e19] Seeds: {seeds}")
    print(f"[e19] Layers: {layers}")
    print(f"[e19] Total cells: {len(seeds) * len(layers)}")
    
    # run all cells
    completed = 0
    skipped = 0
    
    for layer in layers:
        for seed in seeds:
            cell_path = cells_dir / f"L{layer}_s{seed}.json"
            
            if args.skip_existing and cell_path.exists():
                print(f"[e19] SKIP: {cell_path.name} (already exists)")
                skipped += 1
                continue
            
            try:
                result = run_canonical_cell(
                    seed=seed,
                    bank_layer=layer,
                    device=args.device,
                    model_name=args.model,
                    bank_pt=args.bank_pt,
                    n_train=args.n_train,
                    n_eval=args.n_eval,
                    n_preload=args.n_preload,
                    steps=args.steps,
                    lr=args.lr,
                    rank=args.rank,
                )
                
                cell_path.write_text(json.dumps(result, indent=2))
                print(f"[e19] -> {cell_path}")
                completed += 1
                
            except Exception as e:
                print(f"[e19] ERROR in L{layer}_s{seed}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n[e19] Cells completed: {completed}, skipped: {skipped}")
    
    # aggregate summary
    try:
        # import scipy here for aggregation (not needed for cell runs)
        import scipy
    except ImportError:
        print("[e19] WARNING: scipy not available, installing for aggregation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    
    summary = aggregate_summary(cells_dir, layers, seeds)
    summary_path = HERE / "e19_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[e19] Summary -> {summary_path}")
    
    # final verdict
    all_pass = all(summary["layers"].get(str(l), {}).get("pass", False) for l in layers)
    print(f"\n[e19] FINAL: {'PASS' if all_pass else 'FAIL'}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
