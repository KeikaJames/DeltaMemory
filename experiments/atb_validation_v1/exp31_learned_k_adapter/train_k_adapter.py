"""Exp31 Φ2/Φ3 — Train K-adapter on cached InfoNCE pairs.

Reads:  data/paraphrase_cache/<tag>_train.pt, _val.pt
Writes: out_dir/k_projector_seed{seed}.pt  +  train_metrics.json

Validation = retrieval accuracy on val split:
    for each paraphrase q in val, compute
        s_i = mean_layers cos( pool_heads(proj(L, K_i)), pool_heads(q_L) )
    rank correct anchor among val.K; report top-1 / top-5.

Usage:
    python3 .../train_k_adapter.py \\
        --model-tag qwen3-4b --rank 64 \\
        --out-dir run_mps_exp31_qwen_smoke/seed0 \\
        --seed 0 --epochs 50 --batch 128 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deltamemory.memory.k_projector import KProjectorBank, InfoNCEBatch, infonce_loss  # noqa: E402

CACHE_DIR = Path(__file__).parent / "data" / "paraphrase_cache"


def load_cache(tag: str, split: str, device: str):
    p = CACHE_DIR / f"{tag}_{split}.pt"
    d = torch.load(p, weights_only=False, map_location="cpu")
    return d


def build_projector(head_dim: int, n_layers: int, rank: int | None) -> KProjectorBank:
    return KProjectorBank([head_dim] * n_layers, rank=rank)


@torch.no_grad()
def val_retrieval(proj: KProjectorBank, val: dict, device: str) -> dict:
    """Top-1 / top-5 retrieval on val: each paraphrase Q ranks all val anchor K's."""
    anchor_K = val["anchor_K"].to(device)        # [N, L, Hkv, d]
    queries_Q = val["queries_Q"].to(device)       # [N, P, L, Hq, d]
    N, L, Hkv, d = anchor_K.shape
    _, P, _, Hq, _ = queries_Q.shape

    # Project + pool heads + normalize for every layer of every anchor.
    pooled_K = torch.zeros(N, L, d, device=device, dtype=torch.float32)
    for li in range(L):
        pk = proj(li, anchor_K[:, li].float())        # [N, Hkv, d]
        pooled_K[:, li] = nn.functional.normalize(pk.mean(dim=1), dim=-1)

    pooled_Q = nn.functional.normalize(queries_Q.float().mean(dim=-2), dim=-1)  # [N, P, L, d]

    top1 = 0
    top5 = 0
    total = 0
    for p_idx in range(P):
        # mean over layers of cos sim — paraphrases of fact i should retrieve K_i.
        q = pooled_Q[:, p_idx]                        # [N, L, d]
        # sim[n_q, n_k] = mean_L <q[n_q,L], K[n_k,L]>
        sim = torch.einsum("qld,kld->qk", q, pooled_K) / L
        ranks = sim.argsort(dim=-1, descending=True)
        gold = torch.arange(N, device=device)
        top1 += (ranks[:, 0] == gold).sum().item()
        top5 += (ranks[:, :5] == gold.unsqueeze(-1)).any(dim=-1).sum().item()
        total += N
    return {"top1": top1 / total, "top5": top5 / total, "total": total, "N_anchors": N}


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = args.device

    train_d = load_cache(args.model_tag, "train", device)
    val_d = load_cache(args.model_tag, "val", device)

    N_train = train_d["anchor_K"].shape[0]
    L = int(train_d["meta"]["n_layers"])
    Hkv = int(train_d["meta"]["n_kv_heads"])
    Hq = int(train_d["meta"]["n_q_heads"])
    Dh = int(train_d["meta"]["head_dim"])
    P = int(train_d["meta"]["n_paraphrases"])
    print(f"train: N={N_train}  L={L}  Hq={Hq}  Hkv={Hkv}  Dh={Dh}  P={P}")
    print(f"val:   N={val_d['anchor_K'].shape[0]}")

    proj = build_projector(Dh, L, args.rank).to(device).to(torch.float32)
    print(f"projector: rank={args.rank}  params={sum(p.numel() for p in proj.parameters())}")

    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-4)

    # Flatten (fact_idx, paraphrase_idx) pairs for training.
    anchor_K = train_d["anchor_K"]                          # [N, L, Hkv, d]
    queries_Q = train_d["queries_Q"]                        # [N, P, L, Hq, d]
    if getattr(args, "shuffle_pairs", False):
        g = torch.Generator().manual_seed(args.seed + 9999)
        perm = torch.randperm(queries_Q.shape[0], generator=g)
        # ensure no fixed points
        while int((perm == torch.arange(queries_Q.shape[0])).sum().item()) > 0:
            perm = torch.randperm(queries_Q.shape[0], generator=g)
        queries_Q = queries_Q[perm].contiguous()
        print(f"[Gate-E control] queries_Q permuted ({queries_Q.shape[0]} facts)")
    # Per-step we'll sample B fact-pairs (anchor_K[i], queries_Q[i, p~U]).

    metrics = {"epochs": [], "config": vars(args)}
    best_top1 = -1.0
    best_path = Path(args.out_dir) / "k_projector_best.pt"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, N_train // args.batch)
    t0 = time.time()
    for epoch in range(args.epochs):
        proj.train()
        # shuffle fact indices each epoch
        perm = torch.randperm(N_train)
        epoch_loss_layers: list[float] = []
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch : (step + 1) * args.batch]
            if idx.numel() < 2:
                continue
            # random paraphrase pick per fact
            pidx = torch.randint(0, P, (idx.numel(),))
            ak = anchor_K[idx].to(device).float()                          # [B, L, Hkv, d]
            qq = queries_Q[idx, pidx].to(device).float()                   # [B, L, Hq, d]

            # Sum InfoNCE over layers (uniform weighting).
            loss = ak.new_zeros(())
            for li in range(L):
                batch = InfoNCEBatch(layer_idx=li, write_k=ak[:, li], query_q=qq[:, li])
                loss = loss + infonce_loss(proj, batch, temperature=args.tau)
            loss = loss / L

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss_layers.append(float(loss.item()))

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            proj.eval()
            ret = val_retrieval(proj, val_d, device)
            elapsed = time.time() - t0
            chance = 1.0 / ret["N_anchors"]
            print(f"  epoch {epoch+1:3d}  loss={sum(epoch_loss_layers)/max(1,len(epoch_loss_layers)):.4f}  "
                  f"val_top1={ret['top1']:.4f} ({ret['top1']/chance:.2f}× chance)  "
                  f"val_top5={ret['top5']:.4f}  ({elapsed:.0f}s)", flush=True)
            metrics["epochs"].append({
                "epoch": epoch + 1,
                "loss": sum(epoch_loss_layers) / max(1, len(epoch_loss_layers)),
                "val_top1": ret["top1"], "val_top5": ret["top5"],
                "chance": chance, "elapsed_s": elapsed,
            })
            if ret["top1"] > best_top1:
                best_top1 = ret["top1"]
                proj.cpu()
                proj.save(best_path)
                proj.to(device)

    # Always save final state too.
    final_path = Path(args.out_dir) / f"k_projector_seed{args.seed}.pt"
    proj.cpu().save(final_path)
    metrics["best_val_top1"] = best_top1
    metrics["final_path"] = str(final_path)
    metrics["best_path"] = str(best_path)
    (Path(args.out_dir) / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved: {final_path}  best: {best_path}  best_top1={best_top1:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=64, help="low-rank rank (None for full Linear)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--shuffle-pairs", action="store_true",
                    help="(Gate E control) train with paraphrase Q's shuffled across "
                         "fact identities — projector should learn nothing useful.")
    args = ap.parse_args()
    if args.rank is not None and args.rank <= 0:
        args.rank = None
    train(args)
