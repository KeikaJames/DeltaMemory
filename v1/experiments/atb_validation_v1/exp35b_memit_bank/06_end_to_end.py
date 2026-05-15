"""Exp35b — 06: End-to-end Φ3 (router + patch + read), full 1500 test set."""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
sys.path.insert(0, str(EXP35))
_bspec = iu.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = iu.module_from_spec(_bspec); _bspec.loader.exec_module(_bb)
_rspec = iu.spec_from_file_location("exp35_router", EXP35 / "train_router.py")
_tr = iu.module_from_spec(_rspec); _rspec.loader.exec_module(_tr)

first_target_id = _bb.first_target_id
apply_factors = _bb.apply_factors
restore = _bb.restore
margin_at_last = _bb.margin_at_last
assert_bit_equal = _bb.assert_bit_equal
RouterHead = _tr.RouterHead
subject_embed = _tr.subject_embed


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
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--bank", default=str(DATA / "bank.pt"))
    ap.add_argument("--router", default=str(HERE / "router_10k.pt"))
    ap.add_argument("--cache", default=str(DATA / "embeds_cache_10k.pt"))
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b"))
    args = ap.parse_args()

    seed_everything(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    rb = torch.load(args.router, map_location="cpu", weights_only=False)
    id2label = rb["id2label"]; all_ids = rb["all_ids"]
    n_classes = rb["n_classes"]; d_in = rb["d_in"]
    router = RouterHead(d_in, 1024, n_classes)
    router.load_state_dict(rb["router_state"])
    router.eval()

    print(f"[load model] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()

    test_rows_raw = json.load(open(DATA / "splits" / "test.json"))[: args.n_test]
    test_rows = [r for r in test_rows_raw if r["id"] in id2label
                 and not entries[r["id"]].get("norm_outlier", False)]

    cache = torch.load(args.cache, weights_only=False) if Path(args.cache).exists() else None
    cache_index = {fid: i for i, fid in enumerate(cache["ids_test"])} if cache else {}

    rows = []
    t0 = time.time()
    for i, r in enumerate(test_rows):
        fid = r["id"]
        e = entries[fid]
        t_new = first_target_id(tok, e["target_new"])
        t_true = first_target_id(tok, e["target_true"])

        if fid in cache_index:
            emb = cache["X_test"][cache_index[fid]]
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

        base_mean = margins_for(model, tok, r, t_new, t_true)
        assert_bit_equal(model, args.edit_layer, W_ref)

        # routed top-1
        ep = entries[top1_fid]
        W_old = apply_factors(model, args.edit_layer,
                              [(ep["b"].to(device, dtype=dtype),
                                ep["a"].to(device, dtype=dtype))])
        try:
            routed_mean = margins_for(model, tok, r, t_new, t_true)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)

        # oracle (correct fact)
        eo = entries[fid]
        W_old = apply_factors(model, args.edit_layer,
                              [(eo["b"].to(device, dtype=dtype),
                                eo["a"].to(device, dtype=dtype))])
        try:
            oracle_mean = margins_for(model, tok, r, t_new, t_true)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)

        rows.append({
            "fact_id": fid, "router_top1": top1_fid, "router_correct": is_correct,
            "router_top3": top3_fids, "top3_correct": fid in top3_fids,
            "base_mean": base_mean, "routed_mean": routed_mean, "oracle_mean": oracle_mean,
            "routed_uplift": routed_mean - base_mean,
            "oracle_uplift": oracle_mean - base_mean,
        })
        if (i + 1) % 100 == 0:
            recent = rows[-100:]
            print(f"  {i+1}/{len(test_rows)}  routed_uplift={sum(x['routed_uplift'] for x in recent)/len(recent):+.2f}  "
                  f"top1={sum(1 for x in recent if x['router_correct'])/len(recent):.0%}  "
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
            sum(r["routed_uplift"] for r in rows) / max(1e-9, sum(r["oracle_uplift"] for r in rows)),
        "pre_registered_min_ratio": 0.70,
    }
    json.dump(summary, open(out / "phi3_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
