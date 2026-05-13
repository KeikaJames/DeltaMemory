"""Exp35 Φ2 — Learn router (subject embedding -> fact_id) + shuffled baseline.

Subject embedding source (C1, C4, C11):
  - feed prompt through frozen Qwen3
  - locate subject token span
  - take mean of layer-2 hidden states at subject token positions
  - subject embedding is NEVER trained; only a small MLP head learns

Cross-paraphrase split (C3):
  - train inputs : row["prompt"].format(subject)
  - val inputs   : row["paraphrase_prompts"][0]
  - test inputs  : row["paraphrase_prompts"][1]

Anti-cheat C8: train a *shuffled-label* router (random (subj, fact_id)
pairs) with the same architecture and same training budget; its test
top-1 must be ≤ chance + 3SE (≈ 5% for N=975). Otherwise STOP — data
has a statistical bypass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402
from build_bank import subject_last_pos  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "exp31_learned_k_adapter" / "data" / "splits"
EMBED_LAYER = 2


def subject_span(tokenizer, prompt: str, subject: str):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    for variant in (" " + subject.strip(), subject.strip()):
        sids = tokenizer(variant, add_special_tokens=False).input_ids
        if not sids:
            continue
        for i in range(len(ids) - len(sids), -1, -1):
            if ids[i:i + len(sids)] == sids:
                return (i, i + len(sids) - 1)
    # fallback: last token only
    return (len(ids) - 1, len(ids) - 1)


@torch.no_grad()
def subject_embed(model, tok, prompt: str, subject: str):
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[EMBED_LAYER][0]   # (T, d_model)
    lo, hi = subject_span(tok, prompt, subject)
    return h[lo:hi + 1].mean(dim=0).float().cpu()


def collect_embeds(model, tok, rows, prompt_key, with_para_index=None):
    embeds = []
    fids = []
    for i, r in enumerate(rows):
        if prompt_key == "prompt":
            p = r["prompt"].format(r["subject"])
        else:
            paras = r.get("paraphrase_prompts", [])
            if with_para_index is None or with_para_index >= len(paras):
                p = r["prompt"].format(r["subject"])
            else:
                p = paras[with_para_index]
        e = subject_embed(model, tok, p, r["subject"])
        embeds.append(e)
        fids.append(r["id"])
        if (i + 1) % 100 == 0:
            print(f"  embed {i+1}/{len(rows)}", flush=True)
    return torch.stack(embeds), fids


class RouterHead(nn.Module):
    def __init__(self, d_in, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_router(X_train, y_train, X_val, y_val, X_test, y_test,
                 n_classes, *, hidden=1024, lr=1e-3, epochs=50,
                 seed=0, label_name=""):
    torch.manual_seed(seed)
    model = RouterHead(X_train.size(1), hidden, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), 64):
            ix = perm[i:i + 64]
            logits = model(X_train[ix])
            loss = F.cross_entropy(logits, y_train[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(X_val).argmax(dim=-1)
            v_top1 = (v_pred == y_val).float().mean().item()
        if v_top1 > best_val:
            best_val = v_top1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"  [{label_name}] ep{ep+1}  val_top1={v_top1:.3f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_t = model(X_test)
        top1 = (logits_t.argmax(-1) == y_test).float().mean().item()
        top5 = (logits_t.topk(5, dim=-1).indices == y_test.unsqueeze(-1)).any(-1).float().mean().item()
    return model, {"val_top1_best": best_val, "test_top1": top1, "test_top5": top5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-test", type=int, default=125)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=str(HERE / "embeds_cache.pt"))
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35"))
    ap.add_argument("--router-out", default=str(HERE / "router.pt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_rows = json.load(open(SPLITS / "train.json"))
    val_rows = json.load(open(SPLITS / "val.json"))
    test_rows = json.load(open(SPLITS / "test.json"))[: args.n_test]

    # Filter to solo_pass facts in bank (consistent with Φ1)
    bank = torch.load(HERE / "bank.pt", map_location="cpu", weights_only=False)
    ok = {fid for fid, e in bank["entries"].items()
          if e["solo_pass"] and not e.get("norm_outlier", False)}
    train_rows = [r for r in train_rows if r["id"] in ok]
    val_rows = [r for r in val_rows if r["id"] in ok]
    test_rows = [r for r in test_rows if r["id"] in ok]
    print(f"after solo_pass filter: train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}",
          flush=True)

    # Label space = all bank facts (concat order)
    all_ids = list(ok)
    id2label = {fid: i for i, fid in enumerate(all_ids)}
    n_classes = len(all_ids)
    print(f"N_classes (bank size) = {n_classes}", flush=True)

    # Cache embeds (expensive)
    if Path(args.cache).exists():
        print(f"[load] {args.cache}", flush=True)
        cache = torch.load(args.cache, weights_only=False)
    else:
        print(f"[load model] {args.model}", flush=True)
        tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print("[embed train (prompt)]", flush=True)
        X_tr, ids_tr = collect_embeds(model, tok, train_rows, "prompt")
        print("[embed val (paraphrase[0])]", flush=True)
        X_va, ids_va = collect_embeds(model, tok, val_rows, "para", with_para_index=0)
        print("[embed test (paraphrase[1])]", flush=True)
        X_te, ids_te = collect_embeds(model, tok, test_rows, "para", with_para_index=1)
        cache = {"X_train": X_tr, "ids_train": ids_tr,
                 "X_val": X_va, "ids_val": ids_va,
                 "X_test": X_te, "ids_test": ids_te}
        torch.save(cache, args.cache)
        # release model
        del model

    y_train = torch.tensor([id2label[i] for i in cache["ids_train"]])
    y_val = torch.tensor([id2label[i] for i in cache["ids_val"]])
    y_test = torch.tensor([id2label[i] for i in cache["ids_test"]])
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    print(f"X: train={tuple(X_train.shape)} val={tuple(X_val.shape)} test={tuple(X_test.shape)}",
          flush=True)

    # --- HONEST router ---
    print("\n[Φ2] honest router", flush=True)
    t0 = time.time()
    honest, honest_metrics = train_router(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_classes, epochs=args.epochs, lr=args.lr, seed=args.seed,
        label_name="honest",
    )
    print(f"honest: {honest_metrics}  ({time.time()-t0:.0f}s)", flush=True)

    # --- SHUFFLED-LABEL baseline (C8) ---
    print("\n[Φ2] shuffled-label baseline (anti-cheat C8)", flush=True)
    gen = torch.Generator().manual_seed(31415)
    y_train_shuf = y_train[torch.randperm(y_train.size(0), generator=gen)]
    y_val_shuf = y_val[torch.randperm(y_val.size(0), generator=gen)]
    # test labels stay honest — we measure if router learned anything beyond noise.
    t0 = time.time()
    shuf_model, shuf_metrics = train_router(
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
        "pre_registered": {
            "test_top1_min": 0.50,
            "test_top5_min": 0.80,
            "shuffled_baseline_top1_max": 0.05,
        },
        "verdict": {
            "honest_pass_top1": honest_metrics["test_top1"] >= 0.50,
            "honest_pass_top5": honest_metrics["test_top5"] >= 0.80,
            "shuffled_safe": shuf_metrics["test_top1"] <= 0.05,
        },
    }
    json.dump(summary, open(Path(args.out) / "phi2_summary.json", "w"), indent=2)
    torch.save({"router_state": honest.state_dict(),
                "id2label": id2label, "all_ids": all_ids,
                "n_classes": n_classes, "d_in": X_train.size(1)},
               args.router_out)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
