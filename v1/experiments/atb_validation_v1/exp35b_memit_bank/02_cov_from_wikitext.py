"""Exp35b — 02: Estimate MEMIT preconditioner C = E[x x^T] (D8).

x = input to layer L=5 mlp.down_proj (Qwen3-4B), captured at every token
position over a sample of WikiText-103 train tokens. Counterfact subjects
(across the full 10k bank) are masked out at the token level before xx^T
accumulation.

Outputs:
  - exp35b_memit_bank/data/cov_L5.pt  : dict(C, C_inv, n_tokens, sha)
  - exp35b_memit_bank/data/cov_L5_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--target-tokens", type=int, default=2_000_000)
    ap.add_argument("--batch-len", type=int, default=512)
    ap.add_argument("--reg", type=float, default=1e-3)  # τ for C + τ tr(C)/d I
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "cov_L5.pt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build subject mask vocabulary from the 10k corpus.
    corpus = []
    for line in open(DATA / "corpus_10k.jsonl"):
        corpus.append(json.loads(line))
    subjects = set(f["subject"] for f in corpus)
    print(f"[mask] {len(subjects)} unique subjects to mask", flush=True)

    # Build subject token-id n-gram patterns; mask token POSITIONS that fall
    # inside a subject span match (not every position whose id appears in any
    # subject). Single-token subjects masked as token-id set.
    multi_patterns: list = []  # list of tuples of token ids
    single_ids: set = set()
    for s in subjects:
        for variant in (s, " " + s):
            ids = tuple(tok(variant, add_special_tokens=False).input_ids)
            if len(ids) == 0:
                continue
            if len(ids) == 1:
                single_ids.add(int(ids[0]))
            else:
                multi_patterns.append(ids)
    # Deduplicate multi-patterns
    multi_patterns = list({p for p in multi_patterns})
    # Group multi-patterns by first token for fast scanning
    by_first: dict = {}
    for p in multi_patterns:
        by_first.setdefault(int(p[0]), []).append(p)
    print(f"[mask] {len(single_ids)} single-token subj ids; "
          f"{len(multi_patterns)} multi-token subj patterns "
          f"(over {len(by_first)} first-token buckets)", flush=True)

    # Load WikiText-103 train
    from datasets import load_dataset
    print("[load] WikiText-103 train ...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    print(f"[load] {len(ds)} rows", flush=True)

    layer = model.model.layers[args.edit_layer].mlp.down_proj
    d_in = layer.weight.shape[1]
    print(f"[shape] d_intermediate = {d_in}", flush=True)

    # Accumulators on CPU fp32 to avoid MPS fp32 mem pressure on big matmul.
    device = next(model.parameters()).device
    C = torch.zeros((d_in, d_in), dtype=torch.float32, device=device)
    n_used = 0
    n_seen = 0

    captured = {}

    def hook(module, inp, out):
        captured["x"] = inp[0].detach()  # (B, T, d_in)

    h = layer.register_forward_hook(hook)
    t0 = time.time()
    try:
        i_row = 0
        with torch.no_grad():
            while n_used < args.target_tokens and i_row < len(ds):
                text = ds[i_row]["text"]
                i_row += 1
                if not text or len(text) < 20:
                    continue
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.batch_len, add_special_tokens=False).to(device)
                ids = enc["input_ids"]
                if ids.shape[1] < 8:
                    continue
                _ = model(**enc, use_cache=False)
                x = captured["x"][0].float()  # (T, d_in)
                ids_flat = ids[0].tolist()
                # Mask positions inside subject spans (n-gram + single-token)
                mask = [False] * len(ids_flat)
                for i_pos in range(len(ids_flat)):
                    if ids_flat[i_pos] in single_ids:
                        mask[i_pos] = True
                        continue
                    bucket = by_first.get(ids_flat[i_pos])
                    if not bucket:
                        continue
                    for pat in bucket:
                        L = len(pat)
                        if i_pos + L <= len(ids_flat) and tuple(ids_flat[i_pos:i_pos + L]) == pat:
                            for k_off in range(L):
                                mask[i_pos + k_off] = True
                            break
                keep = torch.tensor([not m for m in mask], device=device, dtype=torch.bool)
                x = x[keep]  # (Tk, d_in)
                if x.shape[0] == 0:
                    continue
                C += x.t() @ x  # accumulate xx^T
                n_used += int(x.shape[0])
                n_seen += int(ids.shape[1])
                if (i_row % 200) == 0 or n_used >= args.target_tokens:
                    elapsed = time.time() - t0
                    print(f"  row {i_row}  used={n_used}/{args.target_tokens}  "
                          f"seen={n_seen}  mask_drop={1 - n_used/max(1,n_seen):.2%}  "
                          f"({elapsed:.0f}s)", flush=True)
    finally:
        h.remove()

    print(f"[done] used {n_used} tokens; computing C / n", flush=True)
    C = C / max(1, n_used)
    trC = float(C.diag().sum())
    reg_val = args.reg * trC / d_in
    print(f"[reg] adding {reg_val:.3e} * I (reg={args.reg} * tr(C)/d)", flush=True)
    C_reg = C + reg_val * torch.eye(d_in, device=C.device, dtype=C.dtype)

    # Move to CPU for inversion (bigger memory headroom)
    C_reg_cpu = C_reg.cpu()
    print("[inv] computing C^-1 via Cholesky (CPU fp32) ...", flush=True)
    try:
        L = torch.linalg.cholesky(C_reg_cpu)
        I = torch.eye(d_in, dtype=torch.float32)
        C_inv = torch.cholesky_solve(I, L)
    except Exception as e:
        print(f"[inv] Cholesky failed: {e}; falling back to torch.linalg.inv", flush=True)
        C_inv = torch.linalg.inv(C_reg_cpu)

    # Sanity check: ||C_reg @ C_inv - I||_F
    err = float((C_reg_cpu @ C_inv - torch.eye(d_in)).norm())
    print(f"[inv] ||C@C^-1 - I||_F = {err:.3e}", flush=True)

    out_obj = {
        "C": C_reg_cpu.to(torch.float32),
        "C_inv": C_inv.to(torch.float32),
        "n_tokens_used": n_used,
        "n_tokens_seen": n_seen,
        "edit_layer": args.edit_layer,
        "reg": args.reg,
        "reg_value": reg_val,
        "trace_C": trC,
        "d_in": d_in,
        "inv_residual_norm": err,
    }
    torch.save(out_obj, args.out)

    # Hash the corpus subset implicitly via meta
    meta = {
        "edit_layer": args.edit_layer,
        "n_tokens_used": n_used,
        "n_tokens_seen": n_seen,
        "mask_drop_rate": 1 - n_used / max(1, n_seen),
        "reg": args.reg,
        "reg_value": reg_val,
        "trace_C": trC,
        "d_in": d_in,
        "inv_residual_norm": err,
        "n_masked_token_ids": len(single_ids) + len(multi_patterns),
        "wikitext_source": "wikitext-103-raw-v1/train",
        "elapsed_sec": time.time() - t0,
    }
    json.dump(meta, open(DATA / "cov_L5_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
