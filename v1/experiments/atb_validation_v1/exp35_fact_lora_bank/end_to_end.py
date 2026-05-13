"""Exp35 Φ3 — End-to-end: router selects, oracle patches, read.

For each test fact i:
  1. Compute subject embedding from paraphrase_prompts[1] (cross-paraphrase split).
  2. Router predicts top-1 fact_id (and top-3 for ensemble reference).
  3. Patch W with the predicted fact's (b, a).
  4. Measure target/shuffled margins exactly as Φ1 (k=1 case).

Pass criterion: end-to-end Gate B/D fractions ≥ 70% of Φ1 k=1 oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402
from build_bank import (  # noqa: E402
    first_target_id, apply_factors, restore, margin_at_last, assert_bit_equal,
)
from train_router import RouterHead, subject_embed  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "exp31_learned_k_adapter" / "data" / "splits"


def margins_for(model, tok, row, t_new, t_true):
    prompts = [row["prompt"].format(row["subject"])] + list(row.get("paraphrase_prompts", []))[:2]
    ms = [margin_at_last(model, tok, p, t_new, t_true) for p in prompts]
    return sum(ms) / len(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=125)
    ap.add_argument("--bank", default=str(HERE / "bank.pt"))
    ap.add_argument("--router", default=str(HERE / "router.pt"))
    ap.add_argument("--cache", default=str(HERE / "embeds_cache.pt"))
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35"))
    args = ap.parse_args()

    seed_everything(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    router_blob = torch.load(args.router, map_location="cpu", weights_only=False)
    id2label = router_blob["id2label"]; all_ids = router_blob["all_ids"]
    n_classes = router_blob["n_classes"]; d_in = router_blob["d_in"]
    router = RouterHead(d_in, 1024, n_classes)
    router.load_state_dict(router_blob["router_state"])
    router.eval()

    print(f"[load model] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()

    test_rows_raw = json.load(open(SPLITS / "test.json"))[: args.n_test]
    test_rows = [r for r in test_rows_raw if r["id"] in id2label
                 and entries[r["id"]]["solo_pass"]
                 and not entries[r["id"]].get("norm_outlier", False)]

    # Use cached embeddings if available (cross-paraphrase test split)
    cache = torch.load(args.cache, weights_only=False) if Path(args.cache).exists() else None

    rows = []
    t0 = time.time()
    for i, r in enumerate(test_rows):
        fid = r["id"]
        e = entries[fid]
        t_new = first_target_id(tok, e["target_new"])
        t_true = first_target_id(tok, e["target_true"])

        # Get subject embedding from paraphrase_prompts[1]
        if cache is not None and fid in cache["ids_test"]:
            ix = cache["ids_test"].index(fid)
            emb = cache["X_test"][ix]
        else:
            paras = r.get("paraphrase_prompts", [])
            p = paras[1] if len(paras) > 1 else r["prompt"].format(r["subject"])
            emb = subject_embed(model, tok, p, r["subject"])

        with torch.no_grad():
            logits = router(emb.unsqueeze(0))
            top1_idx = int(logits.argmax(-1).item())
            top3_idx = logits.topk(3, dim=-1).indices[0].tolist()

        top1_fid = all_ids[top1_idx]
        top3_fids = [all_ids[j] for j in top3_idx]
        is_correct = (top1_fid == fid)

        # base margin
        base_mean = margins_for(model, tok, r, t_new, t_true)
        assert_bit_equal(model, args.edit_layer, W_ref)

        # patch with router top-1 selection
        e_pred = entries[top1_fid]
        b = e_pred["b"].to(device, dtype=dtype)
        a = e_pred["a"].to(device, dtype=dtype)
        W_old = apply_factors(model, args.edit_layer, [(b, a)])
        try:
            routed_mean = margins_for(model, tok, r, t_new, t_true)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)

        # also: oracle (correct fact) for comparison
        b_o = entries[fid]["b"].to(device, dtype=dtype)
        a_o = entries[fid]["a"].to(device, dtype=dtype)
        W_old = apply_factors(model, args.edit_layer, [(b_o, a_o)])
        try:
            oracle_mean = margins_for(model, tok, r, t_new, t_true)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)

        rows.append({
            "fact_id": fid, "router_top1": top1_fid, "router_correct": is_correct,
            "router_top3": top3_fids, "top3_correct": fid in top3_fids,
            "base_mean": base_mean,
            "routed_mean": routed_mean,
            "oracle_mean": oracle_mean,
            "routed_uplift": routed_mean - base_mean,
            "oracle_uplift": oracle_mean - base_mean,
        })
        if (i + 1) % 25 == 0:
            recent = rows[-25:]
            print(f"  {i+1}/{len(test_rows)}  routed_uplift="
                  f"{sum(r['routed_uplift'] for r in recent)/25:+.2f}  "
                  f"router_top1={sum(1 for r in recent if r['router_correct'])/25:.0%}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open(out / "phi3_cells.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    summary = {
        "n_test": n,
        "router_top1_acc": sum(1 for r in rows if r["router_correct"]) / n,
        "router_top3_acc": sum(1 for r in rows if r["top3_correct"]) / n,
        "mean_routed_uplift": sum(r["routed_uplift"] for r in rows) / n,
        "mean_oracle_uplift": sum(r["oracle_uplift"] for r in rows) / n,
        "frac_routed_beats_base": sum(1 for r in rows if r["routed_uplift"] > 0) / n,
        "frac_oracle_beats_base": sum(1 for r in rows if r["oracle_uplift"] > 0) / n,
        "learned_vs_oracle_ratio_uplift":
            (sum(r["routed_uplift"] for r in rows) / max(1e-9, sum(r["oracle_uplift"] for r in rows))),
        "pre_registered_min_ratio": 0.70,
    }
    json.dump(summary, open(out / "phi3_summary.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
