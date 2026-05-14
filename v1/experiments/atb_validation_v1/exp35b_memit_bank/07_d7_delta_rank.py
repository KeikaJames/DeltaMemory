"""Exp35b — 07: D7 Δ-rank audit.

Stack all bank Δ_i = b_i a_iᵀ outer-products into a single matrix
M = Σ_i b_i a_iᵀ ∈ R^(d_out × d_in) — NO, this loses fact identity.
Instead, compute the effective rank of {b_i a_iᵀ} viewed as a set:
  - Stack columns to form B ∈ R^(d_out × N), A ∈ R^(d_in × N).
  - The space spanned by all individual rank-1 patches has dimension
    rank(B) · rank(A) in worst case (≤ N).
  - Effective rank via singular values entropy.

Outputs:
  - SVD spectrum of B and A (top 100 singular values + tail energy)
  - Effective rank (entropy of normalised sv²)
  - Cosine similarity matrix histogram of {a_i / ‖a_i‖} (D7: are keys
    distinguishable?)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(DATA / "bank.pt"))
    ap.add_argument("--n-sample", type=int, default=2000,
                    help="random sample size for cosine-sim histogram")
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    fids = list(entries.keys())
    N = len(fids)
    print(f"[load] {N} facts", flush=True)

    e0 = entries[fids[0]]
    d_out, d_in = e0["b"].shape[0], e0["a"].shape[0]
    print(f"d_out={d_out} d_in={d_in}", flush=True)

    # Stack
    B = torch.stack([entries[f]["b"].float() for f in fids], dim=1)  # (d_out, N)
    A = torch.stack([entries[f]["a"].float() for f in fids], dim=1)  # (d_in, N)
    norms = torch.tensor([entries[f]["delta_norm"] for f in fids]).float()

    # SVD
    print("[svd B] ...", flush=True)
    sB = torch.linalg.svdvals(B)
    print("[svd A] ...", flush=True)
    sA = torch.linalg.svdvals(A)

    def effective_rank(s):
        p = (s ** 2) / (s ** 2).sum()
        p = p[p > 0]
        H = -(p * p.log()).sum()
        return float(H.exp())

    er_B = effective_rank(sB)
    er_A = effective_rank(sA)
    print(f"eff_rank(B)={er_B:.1f}  eff_rank(A)={er_A:.1f}  (N={N})", flush=True)

    # Cosine sim distribution of a-keys
    torch.manual_seed(0)
    idx = torch.randperm(N)[: min(args.n_sample, N)]
    A_s = A[:, idx]
    A_n = A_s / (A_s.norm(dim=0, keepdim=True) + 1e-8)
    print("[cos sim] computing ...", flush=True)
    cs = (A_n.T @ A_n)
    iu = torch.triu_indices(cs.size(0), cs.size(1), offset=1)
    cs_flat = cs[iu[0], iu[1]]
    cs_abs = cs_flat.abs()

    def pctl(t, q):
        return float(t.kthvalue(int(q * t.numel())).values.item())

    summary = {
        "N": N, "d_out": d_out, "d_in": d_in,
        "delta_norm_stats": {
            "mean": float(norms.mean()),
            "median": float(norms.median()),
            "p99": pctl(norms, 0.99),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "B_singular_top10": sB[:10].tolist(),
        "B_singular_tail_energy_frac": float((sB[100:] ** 2).sum() / (sB ** 2).sum()) if sB.numel() > 100 else None,
        "A_singular_top10": sA[:10].tolist(),
        "A_singular_tail_energy_frac": float((sA[100:] ** 2).sum() / (sA ** 2).sum()) if sA.numel() > 100 else None,
        "effective_rank_B": er_B,
        "effective_rank_A": er_A,
        "cosine_a_keys_sample_size": int(idx.numel()),
        "cosine_a_keys_abs_median": float(cs_abs.median()),
        "cosine_a_keys_abs_p90": pctl(cs_abs, 0.9),
        "cosine_a_keys_abs_p99": pctl(cs_abs, 0.99),
        "frac_high_collinear_above_0p9": float((cs_abs > 0.9).float().mean()),
    }
    json.dump(summary, open(out / "d7_delta_rank.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
