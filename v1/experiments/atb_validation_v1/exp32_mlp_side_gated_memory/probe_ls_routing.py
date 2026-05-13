"""Closed-form least-squares K-routing probe.

Replaces InfoNCE training with closed-form solutions to test whether
the shortcut signal observed under InfoNCE (shuffled-pair test top-1 ≈ 45%)
is a geometric property of the captured (K, Q) activations or an artefact
of contrastive-loss optimisation dynamics on small data.

Two estimators:

1. Full-rank ridge regression per layer
        W_l = argmin_W  Σ ||Q[n,p,l] W − K[n,l]||² + λ ||W||²_F
2. Rank-r CCA per layer (closed-form analogue of InfoNCE key_dim=r head)
        max corr( Q[:,p,l] U_q[l] , K[:,l] U_k[l] )

Both are evaluated on
    (a) real (anchor_K, queries_Q) pairs
    (b) shuffled pairs  (Gate-E LS analogue)
    (c) random-label seeds (3×) for variance baseline.

The expectation under each hypothesis is:

  H_data-shortcut : real and shuffled both score high — small-N geometry
                    contains the shortcut.
  H_optim-shortcut: real scores moderately, shuffled scores at chance —
                    InfoNCE's 45% shuffled signal is dynamics-induced.

Usage:
  python probe_ls_routing.py \
      --cache data/cache/Qwen_Qwen3-4B-Instruct-2507 \
      --out   run_qwen_full/ls_probe.json

Reads the same (anchor_K, queries_Q) cache as train_mlp_gate.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_split(cache_prefix: Path, split: str):
    return torch.load(cache_prefix.with_name(cache_prefix.name + f"_{split}.pt"),
                      map_location="cpu", weights_only=False)


def solve_ridge_per_layer(K: torch.Tensor, Q: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form W_l ∈ R^{D×D} per layer with ridge penalty lam."""
    N, L, D = K.shape
    P = Q.shape[1]
    Q_stack = Q.reshape(-1, L, D)
    K_rep = K.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, L, D)
    Ws = []
    eye = torch.eye(D, dtype=K.dtype)
    for l in range(L):
        Ql, Kl = Q_stack[:, l], K_rep[:, l]
        A = Ql.T @ Ql + lam * eye
        B = Ql.T @ Kl
        Ws.append(torch.linalg.solve(A, B))
    return torch.stack(Ws)


def fit_cca_per_layer(K: torch.Tensor, Q: torch.Tensor, r: int, reg: float):
    """Closed-form rank-r CCA per layer."""
    N, L, D = K.shape
    P = Q.shape[1]
    Q_stack = Q.reshape(-1, L, D)
    K_rep = K.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, L, D)
    Uqs, Uks, fails = [], [], 0
    eye = torch.eye(D, dtype=K.dtype)
    for l in range(L):
        Ql, Kl = Q_stack[:, l], K_rep[:, l]
        Ql_c = Ql - Ql.mean(0, keepdim=True)
        Kl_c = Kl - Kl.mean(0, keepdim=True)
        n = Ql.shape[0]
        Cqq = (Ql_c.T @ Ql_c) / (n - 1) + reg * eye
        Ckk = (Kl_c.T @ Kl_c) / (n - 1) + reg * eye
        Cqk = (Ql_c.T @ Kl_c) / (n - 1)
        try:
            Lq = torch.linalg.cholesky(Cqq)
            Lk = torch.linalg.cholesky(Ckk)
            Wq_inv = torch.linalg.solve_triangular(Lq, eye, upper=False)
            Wk_inv = torch.linalg.solve_triangular(Lk, eye, upper=False)
            M = Wq_inv @ Cqk @ Wk_inv.T
            U, S, Vh = torch.linalg.svd(M + 1e-6 * eye, full_matrices=False)
            Uqs.append(Wq_inv.T @ U[:, :r])
            Uks.append(Wk_inv.T @ Vh.T[:, :r])
        except Exception:
            fails += 1
            Uqs.append(torch.zeros(D, r, dtype=K.dtype))
            Uks.append(torch.zeros(D, r, dtype=K.dtype))
    return torch.stack(Uqs), torch.stack(Uks), fails


def topk_ridge(W: torch.Tensor, K_anchor: torch.Tensor, Q: torch.Tensor):
    K_n = F.normalize(K_anchor, dim=-1)
    hit = tot = top5 = 0
    for p in range(Q.shape[1]):
        proj = F.normalize(torch.einsum("nld,ldD->nlD", Q[:, p], W), dim=-1)
        sims = torch.einsum("nld,mld->nlm", proj, K_n).mean(dim=1)
        top1 = sims.argmax(dim=-1)
        top5_idx = sims.topk(5, dim=-1).indices
        for i in range(top1.shape[0]):
            tot += 1
            hit += int(top1[i].item() == i)
            top5 += int(i in top5_idx[i].tolist())
    return hit / tot, top5 / tot


def topk_cca(Uq: torch.Tensor, Uk: torch.Tensor, K_anchor: torch.Tensor, Q: torch.Tensor):
    K_proj = F.normalize(torch.einsum("nld,ldr->nlr", K_anchor, Uk), dim=-1)
    hit = tot = top5 = 0
    for p in range(Q.shape[1]):
        Q_proj = F.normalize(torch.einsum("nld,ldr->nlr", Q[:, p], Uq), dim=-1)
        sims = torch.einsum("nlr,mlr->nlm", Q_proj, K_proj).mean(dim=1)
        top1 = sims.argmax(dim=-1)
        top5_idx = sims.topk(5, dim=-1).indices
        for i in range(top1.shape[0]):
            tot += 1
            hit += int(top1[i].item() == i)
            top5 += int(i in top5_idx[i].tolist())
    return hit / tot, top5 / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="path prefix, e.g. data/cache/Qwen_Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cache = Path(args.cache)

    train = load_split(cache, "train")
    val = load_split(cache, "val")
    test = load_split(cache, "test")

    Ktr, Qtr = train["anchor_K"].float(), train["queries_Q"].float()
    Kv,  Qv  = val["anchor_K"].float(),   val["queries_Q"].float()
    Kt,  Qt  = test["anchor_K"].float(),  test["queries_Q"].float()
    N, L, D = Ktr.shape
    n_test = Kt.shape[0]
    chance = 1.0 / n_test

    print(f"shapes: train N={N} L={L} D={D}  test={n_test}  chance={chance*100:.2f}%")

    results = {"meta": {"N_train": N, "L": L, "D": D, "N_test": n_test,
                        "chance": chance}, "trials": []}

    # --- Full-rank ridge sweep, real pairs
    print("\n== Full-rank ridge (real pairs) ==")
    print(f"{'lam':>10} {'train_t1':>9} {'val_t1':>9} {'test_t1':>9} {'test_t5':>9}")
    for lam in [1e-1, 1e0, 1e1, 1e2, 1e3]:
        W = solve_ridge_per_layer(Ktr, Qtr, lam)
        tr, _ = topk_ridge(W, Ktr[: Kv.shape[0]], Qtr[: Kv.shape[0]])
        v, _ = topk_ridge(W, Kv, Qv)
        t1, t5 = topk_ridge(W, Kt, Qt)
        print(f"{lam:10.0e} {tr*100:8.2f}% {v*100:8.2f}% {t1*100:8.2f}% {t5*100:8.2f}%")
        results["trials"].append({"method": "ridge", "pairs": "real", "lam": lam,
                                  "train_t1": tr, "val_t1": v, "test_t1": t1, "test_t5": t5})

    # --- CCA real pairs
    print("\n== Rank-r CCA (real pairs, reg=1e-2) ==")
    print(f"{'rank':>6} {'train_t1':>9} {'val_t1':>9} {'test_t1':>9} {'test_t5':>9}")
    for r in [16, 64, 256]:
        Uq, Uk, fails = fit_cca_per_layer(Ktr, Qtr, r, 1e-2)
        tr, _ = topk_cca(Uq, Uk, Ktr[: Kv.shape[0]], Qtr[: Kv.shape[0]])
        v, _ = topk_cca(Uq, Uk, Kv, Qv)
        t1, t5 = topk_cca(Uq, Uk, Kt, Qt)
        print(f"{r:6d} {tr*100:8.2f}% {v*100:8.2f}% {t1*100:8.2f}% {t5*100:8.2f}%  (fails={fails})")
        results["trials"].append({"method": "cca", "pairs": "real", "rank": r, "reg": 1e-2,
                                  "train_t1": tr, "val_t1": v, "test_t1": t1, "test_t5": t5,
                                  "fails": fails})

    # --- CCA shuffled pairs
    print("\n== Rank-64 CCA (shuffled pairs, reg=1e-2) ==")
    g = torch.Generator().manual_seed(1000)
    perm = torch.randperm(N, generator=g)
    Qtr_sh = Qtr[perm]
    Uq, Uk, fails = fit_cca_per_layer(Ktr, Qtr_sh, 64, 1e-2)
    tr, _ = topk_cca(Uq, Uk, Ktr[: Kv.shape[0]], Qtr[: Kv.shape[0]])
    v, _ = topk_cca(Uq, Uk, Kv, Qv)
    t1, t5 = topk_cca(Uq, Uk, Kt, Qt)
    print(f"shuffled: train={tr*100:.2f}% val={v*100:.2f}% test_t1={t1*100:.2f}% test_t5={t5*100:.2f}%  (fails={fails})")
    results["trials"].append({"method": "cca", "pairs": "shuffled", "rank": 64, "reg": 1e-2,
                              "train_t1": tr, "val_t1": v, "test_t1": t1, "test_t5": t5,
                              "fails": fails})

    # --- random-label seeds
    print("\n== Rank-64 CCA (random-label seeds, 3×) ==")
    random_t1 = []
    for seed in (42, 43, 44):
        g = torch.Generator().manual_seed(seed)
        Qtr_r = Qtr[torch.randperm(N, generator=g)]
        Uq, Uk, fails = fit_cca_per_layer(Ktr, Qtr_r, 64, 1e-2)
        t1, _ = topk_cca(Uq, Uk, Kt, Qt)
        random_t1.append(t1)
        print(f"  seed={seed}: test_t1={t1*100:.2f}%  (fails={fails})")
        results["trials"].append({"method": "cca", "pairs": f"random_{seed}", "rank": 64,
                                  "reg": 1e-2, "test_t1": t1, "fails": fails})
    print(f"  mean={sum(random_t1)/3*100:.2f}%  (chance={chance*100:.2f}%)")
    results["meta"]["random_label_mean_test_t1"] = sum(random_t1) / 3

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
