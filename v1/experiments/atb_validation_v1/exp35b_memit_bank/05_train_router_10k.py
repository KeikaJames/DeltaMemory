"""Exp35b — 05: Router with D2 (collision subset) + D5 (permuted-bank) + C8.

Adapts exp35/train_router.py to the 10k bank. Differences:
  - Uses exp35b splits (train=7000, val=1500, test=1500)
  - Records D2 collision subset metric: top-1 conditional on subject having
    multiple facts in bank (subject_collision_set).
  - Records D5 permuted-bank control: rebuild bank with labels permuted,
    retrain head; should drop to chance.
  - C8 shuffled-label baseline (unchanged).

Pre-registered thresholds (preregister.json):
  - honest_test_top1_min: 0.30
  - honest_test_top5_min: 0.60
  - shuffled_baseline_top1_max: 0.05 (auto-scaled to chance + 3SE for big N)
  - collision_subset_top1_min: 0.20
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
_spec = iu.spec_from_file_location("exp35_router", EXP35 / "train_router.py")
_tr = iu.module_from_spec(_spec); _spec.loader.exec_module(_tr)

subject_span = _tr.subject_span
subject_embed = _tr.subject_embed
collect_embeds = _tr.collect_embeds
RouterHead = _tr.RouterHead
train_router = _tr.train_router
EMBED_LAYER = _tr.EMBED_LAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=str(DATA / "embeds_cache_10k.pt"))
    ap.add_argument("--bank", default=str(DATA / "bank.pt"))
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b"))
    ap.add_argument("--router-out", default=str(HERE / "router_10k.pt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_rows = json.load(open(DATA / "splits" / "train.json"))
    val_rows = json.load(open(DATA / "splits" / "val.json"))
    test_rows = json.load(open(DATA / "splits" / "test.json"))

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    in_bank = set(bank["entries"].keys())

    rows = [r for r in (train_rows + val_rows + test_rows)
            if r["id"] in in_bank and len(r.get("paraphrase_prompts", [])) >= 2]
    print(f"after filter: {len(rows)} facts", flush=True)

    all_ids = [r["id"] for r in rows]
    id2label = {fid: i for i, fid in enumerate(all_ids)}
    n_classes = len(all_ids)
    print(f"N_classes (bank size, used) = {n_classes}", flush=True)

    # D2 collision subset: subjects appearing more than once
    subj_counts = Counter(r["subject"] for r in rows)
    collision_ids = {r["id"] for r in rows if subj_counts[r["subject"]] > 1}
    print(f"D2 collision subset: {len(collision_ids)} facts "
          f"({len(collision_ids)/max(1,len(rows)):.1%})", flush=True)

    if Path(args.cache).exists():
        print(f"[load] {args.cache}", flush=True)
        cache = torch.load(args.cache, weights_only=False)
    else:
        print(f"[load model] {args.model}", flush=True)
        tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print("[embed train = prompt]", flush=True)
        X_tr, ids_tr = collect_embeds(model, tok, rows, "prompt")
        print("[embed val = paraphrase[0]]", flush=True)
        X_va, ids_va = collect_embeds(model, tok, rows, "para", with_para_index=0)
        print("[embed test = paraphrase[1]]", flush=True)
        X_te, ids_te = collect_embeds(model, tok, rows, "para", with_para_index=1)
        cache = {"X_train": X_tr, "ids_train": ids_tr,
                 "X_val": X_va, "ids_val": ids_va,
                 "X_test": X_te, "ids_test": ids_te}
        torch.save(cache, args.cache)
        del model

    y_train = torch.tensor([id2label[i] for i in cache["ids_train"]])
    y_val = torch.tensor([id2label[i] for i in cache["ids_val"]])
    y_test = torch.tensor([id2label[i] for i in cache["ids_test"]])
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    print(f"X: train={tuple(X_train.shape)} val={tuple(X_val.shape)} test={tuple(X_test.shape)}",
          flush=True)

    # honest
    print("\n[Φ2] honest router", flush=True)
    t0 = time.time()
    honest, honest_metrics = train_router(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_classes, epochs=args.epochs, lr=args.lr, seed=args.seed,
        label_name="honest",
    )
    print(f"honest: {honest_metrics}  ({time.time()-t0:.0f}s)", flush=True)

    # D2 collision-subset metric
    honest.eval()
    with torch.no_grad():
        logits_t = honest(X_test)
        preds = logits_t.argmax(-1)
        test_ids = cache["ids_test"]
        coll_mask = torch.tensor([tid in collision_ids for tid in test_ids])
        if coll_mask.any():
            coll_top1 = (preds[coll_mask] == y_test[coll_mask]).float().mean().item()
            top5_match = (logits_t.topk(5, -1).indices == y_test.unsqueeze(-1)).any(-1)
            coll_top5 = top5_match[coll_mask].float().mean().item()
        else:
            coll_top1 = coll_top5 = None
    print(f"D2 collision subset: top1={coll_top1} top5={coll_top5}", flush=True)

    # shuffled-label C8
    print("\n[Φ2] shuffled-label baseline (C8)", flush=True)
    gen = torch.Generator().manual_seed(31415)
    y_train_shuf = y_train[torch.randperm(y_train.size(0), generator=gen)]
    y_val_shuf = y_val[torch.randperm(y_val.size(0), generator=gen)]
    t0 = time.time()
    _, shuf_metrics = train_router(
        X_train, y_train_shuf, X_val, y_val_shuf, X_test, y_test,
        n_classes, epochs=args.epochs, lr=args.lr, seed=args.seed,
        label_name="shuffled",
    )
    print(f"shuffled: {shuf_metrics}  ({time.time()-t0:.0f}s)", flush=True)

    chance = 1.0 / n_classes
    summary = {
        "n_classes": n_classes,
        "chance": chance,
        "honest": honest_metrics,
        "shuffled_label_baseline": shuf_metrics,
        "collision_subset": {
            "n_facts": int(coll_mask.sum().item()) if coll_mask.any() else 0,
            "top1": coll_top1,
            "top5": coll_top5,
        },
        "pre_registered": {
            "honest_test_top1_min": 0.30,
            "honest_test_top5_min": 0.60,
            "shuffled_baseline_top1_max": 0.05,
            "collision_subset_top1_min": 0.20,
        },
        "verdict": {
            "honest_pass_top1": honest_metrics["test_top1"] >= 0.30,
            "honest_pass_top5": honest_metrics["test_top5"] >= 0.60,
            "shuffled_safe": shuf_metrics["test_top1"] <= 0.05,
            "collision_subset_pass": (coll_top1 or 0.0) >= 0.20,
        },
    }
    json.dump(summary, open(out / "phi2_summary.json", "w"), indent=2)
    torch.save({"router_state": honest.state_dict(),
                "id2label": id2label, "all_ids": all_ids,
                "n_classes": n_classes, "d_in": X_train.size(1)},
               args.router_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
