"""Exp32 Φ2 — Train MLPGatedRouter (W_q, W_g) on captured (K, V, Q) pairs.

Two-stage loss:
  L_retrieval = InfoNCE on softmax( (W_q Q) @ (W_q K)^T / τ )   per layer, summed
  L_value     = MSE( gate * sum_n softmax(...)_n * V_n , V_anchor[i] )  per layer
  L_total     = L_retrieval + λ_v * L_value

This is purely *embedding-space* training — no base model forward, no autograd
through the LM. It produces a (W_q, W_g) such that:
  1. routing scores peak on the correct fact (Gate B)
  2. the gated readout reconstructs the anchor V vector (Gate C surrogate)

Run:
  python3 experiments/atb_validation_v1/exp32_mlp_side_gated_memory/train_mlp_gate.py \\
    --cache experiments/atb_validation_v1/exp32_mlp_side_gated_memory/data/cache/Qwen_Qwen3-4B-Instruct-2507 \\
    --out  experiments/atb_validation_v1/exp32_mlp_side_gated_memory/run_qwen_smoke/seed0 \\
    --seed 0 --key-dim 64 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from deltamemory.memory.mlp_gated_injector import MLPGatedRouter  # noqa: E402


def load_cache(prefix: Path, split: str):
    pt = torch.load(prefix.with_name(prefix.name + f"_{split}.pt"), map_location="cpu",
                    weights_only=False)
    return pt


def val_retrieval(router: MLPGatedRouter, K_anchor: torch.Tensor, Q: torch.Tensor,
                  device: str = "cpu") -> tuple[float, float]:
    """K_anchor: (N, L, D). Q: (N, P, L, D). Return (top1, top5) avg over paraphrases."""
    router = router.to(device)
    K_anchor = K_anchor.to(device).float()
    Q = Q.to(device).float()
    N, L, D = K_anchor.shape
    P = Q.shape[1]
    # per-layer project + L2 then mean over layers
    mk_proj = []
    for l in range(L):
        mk_proj.append(F.normalize(router.W_q[l](K_anchor[:, l]), dim=-1))
    mk = torch.stack(mk_proj, dim=1)  # (N, L, key_dim)
    hit1 = 0
    hit5 = 0
    total = 0
    for p in range(P):
        mq_proj = []
        for l in range(L):
            mq_proj.append(F.normalize(router.W_q[l](Q[:, p, l]), dim=-1))
        mq = torch.stack(mq_proj, dim=1)  # (N, L, key_dim)
        # mean-over-layers similarity
        sims = torch.einsum("nlk,mlk->nm", mq, mk) / L
        top5 = sims.topk(min(5, N), dim=-1).indices
        for i in range(N):
            total += 1
            if top5[i, 0].item() == i:
                hit1 += 1
            if i in top5[i].tolist():
                hit5 += 1
    return hit1 / max(total, 1), hit5 / max(total, 1)


def train(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    prefix = Path(args.cache)
    train_pt = load_cache(prefix, "train")
    val_pt = load_cache(prefix, "val")

    K_train = train_pt["anchor_K"]   # (N, L, D)
    V_train = train_pt["anchor_V"]
    Q_train = train_pt["queries_Q"]  # (N, P, L, D)
    if getattr(args, "shuffle_pairs", False):
        # Gate E control: scramble fact identity by permuting Q rows so that
        # paraphrases of fact i now serve as positives for a different anchor.
        g = torch.Generator().manual_seed(args.seed + 1000)
        perm = torch.randperm(K_train.shape[0], generator=g)
        Q_train = Q_train[perm].contiguous()
        print(f"[shuffle_pairs] permuted Q rows under seed={args.seed+1000}")
    K_val = val_pt["anchor_K"]
    Q_val = val_pt["queries_Q"]

    N_tr, L, D = K_train.shape
    P = Q_train.shape[1]
    print(f"train N={N_tr} L={L} D={D} P={P}")

    router = MLPGatedRouter(num_layers=L, d_model=D, key_dim=args.key_dim).to(args.device)
    opt = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)

    K_tr = K_train.float().to(args.device)
    V_tr = V_train.float().to(args.device)
    Q_tr = Q_train.float().to(args.device)

    metrics = {"per_epoch": []}
    best_top1 = 0.0
    for ep in range(args.epochs):
        router.train()
        t0 = time.time()
        # shuffle anchor order
        perm = torch.randperm(N_tr, device=args.device)
        K_tr_p = K_tr[perm]
        V_tr_p = V_tr[perm]
        Q_tr_p = Q_tr[perm]
        loss_ret_total = 0.0
        loss_v_total = 0.0
        # full-batch (N≈567 fits on MPS comfortably)
        # For each layer: project anchors and queries, softmax cross-entropy
        ret_sum = torch.zeros((), device=args.device)
        v_sum = torch.zeros((), device=args.device)
        targets = torch.arange(N_tr, device=args.device)
        for l in range(L):
            mk = F.normalize(router.W_q[l](K_tr_p[:, l]), dim=-1)        # (N, key_dim)
            # randomly pick one paraphrase per anchor each epoch
            pidx = torch.randint(0, P, (N_tr,), device=args.device)
            q_in = Q_tr_p[torch.arange(N_tr, device=args.device), pidx, l]
            mq = F.normalize(router.W_q[l](q_in), dim=-1)               # (N, key_dim)
            logits = (mq @ mk.T) / args.temperature
            ret_sum = ret_sum + F.cross_entropy(logits, targets)
            # value reconstruction via softmax(mq@mk^T) @ V
            with torch.no_grad():
                tgt_v = V_tr_p[:, l]
            w = F.softmax(logits, dim=-1)                                # (N, N)
            v_pred = w @ V_tr_p[:, l]                                    # (N, D)
            # learned gate at the query side (W_g uses MLP-input space, i.e. q_in)
            gate = torch.sigmoid(router.W_g[l](q_in)).squeeze(-1)        # (N,)
            v_sum = v_sum + F.mse_loss(gate.unsqueeze(-1) * v_pred, tgt_v)
        loss = ret_sum + args.lambda_v * v_sum
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()
        loss_ret_total = float(ret_sum.item())
        loss_v_total = float(v_sum.item())

        # validation
        router.eval()
        with torch.no_grad():
            top1, top5 = val_retrieval(router, K_val, Q_val, device=args.device)
        dt = time.time() - t0
        print(f"  ep {ep+1:3d}/{args.epochs}  Lret={loss_ret_total:.3f}  Lv={loss_v_total:.4f}  "
              f"val_top1={top1*100:.2f}%  val_top5={top5*100:.2f}%  ({dt:.1f}s)", flush=True)
        metrics["per_epoch"].append({"epoch": ep+1, "L_ret": loss_ret_total,
                                     "L_v": loss_v_total, "val_top1": top1, "val_top5": top5})
        if top1 > best_top1:
            best_top1 = top1
            torch.save({"state_dict": router.state_dict(),
                        "key_dim": args.key_dim, "d_model": D, "n_layers": L,
                        "epoch": ep+1, "val_top1": top1},
                       out / "router_best.pt")

    torch.save({"state_dict": router.state_dict(),
                "key_dim": args.key_dim, "d_model": D, "n_layers": L},
               out / f"router_seed{args.seed}.pt")
    (out / "train_metrics.json").write_text(json.dumps({
        "args": vars(args), "best_val_top1": best_top1, "epochs": args.epochs,
        "per_epoch": metrics["per_epoch"],
    }, indent=2))
    print(f"\n[done] best_val_top1={best_top1*100:.2f}% — saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="prefix path, e.g. .../cache/Qwen_Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lambda-v", type=float, default=0.1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--shuffle-pairs", action="store_true",
                    help="Gate E control: permute Q rows so adapter trains on wrong pairs")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
