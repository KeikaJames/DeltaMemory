"""e03 — capability drift on WikiText-2 / WikiText-103 valid.

After Phase B2 / e02 training, does the projector + bank harm general
language modeling on text unrelated to the bank topics?

Compare four configs, all on the same N tokens of WikiText:
    base       — vanilla model, no LPL, no bank
    bank_off   — LPL patched but bank empty (sanity: must == base)
    bank_on    — bank preloaded with B2-style 512 b-vectors at layer 9
    bank_on_p  — bank_on + a trained projector loaded from a checkpoint

Pass criterion: rel PPL drift |bank_on - base| / base ≤ 5%.

We use a small slice of WikiText-2 (loaded from `datasets`) by default
because WikiText-103 valid is ~250K tokens; flip --tokens to scale.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, data_io,
)


def load_wikitext(tok, n_tokens: int, name: str = "wikitext-2-raw-v1") -> torch.Tensor:
    """Concat valid split text into a single token tensor of length n_tokens."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")
    ds = load_dataset("Salesforce/wikitext", name, split="validation")
    texts = [t for t in ds["text"] if t and not t.isspace()]
    big = "\n".join(texts)
    ids = tok(big, return_tensors="pt").input_ids[0]
    if ids.numel() > n_tokens:
        ids = ids[:n_tokens]
    print(f"[e03] wikitext tokens={ids.numel()} (cap {n_tokens})")
    return ids


def chunked_nll(model, ids: torch.Tensor, *, ctx: int, device: str,
                run_forward) -> float:
    """Sliding-window perplexity across `ids` in non-overlapping chunks of `ctx`."""
    n = ids.numel()
    nlls, n_pred = [], 0
    t0 = time.time()
    for s in range(0, n - 1, ctx):
        e = min(s + ctx, n)
        chunk = ids[s:e].unsqueeze(0).to(device)
        if chunk.shape[1] < 2: continue
        with torch.no_grad():
            logits = run_forward(chunk)
        # next-token NLL
        pred = logits[0, :-1, :].float()
        gold = chunk[0, 1:]
        loss = F.cross_entropy(pred, gold, reduction="sum")
        nlls.append(loss.item()); n_pred += gold.numel()
    elapsed = time.time() - t0
    avg = sum(nlls) / max(n_pred, 1)
    print(f"[e03] tokens={n_pred} avg_nll={avg:.4f} ppl={math.exp(avg):.3f} ({elapsed:.1f}s)")
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tokens", type=int, default=20000)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--projector_ckpt", default=None,
                   help="optional path to a saved P state_dict from e01/e02")
    args = p.parse_args()

    out_path = HERE / f"e03_drift_t{args.tokens}.json"

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters(): pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    ids = load_wikitext(tok, args.tokens)

    # === config 1: base (no patch) ===
    def fwd_base(chunk):
        return model(input_ids=chunk, use_cache=False).logits
    nll_base = chunked_nll(model, ids, ctx=args.ctx, device=args.device, run_forward=fwd_base)

    # === install patch for the rest ===
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    def fwd_lpl(chunk):
        s1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, s1):
            model(input_ids=chunk, use_cache=False)
        s2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, s2):
            return model(input_ids=chunk, use_cache=False).logits

    # === config 2: bank_off (LPL patched, bank empty) ===
    bank.clear()
    nll_off = chunked_nll(model, ids, ctx=args.ctx, device=args.device, run_forward=fwd_lpl)

    # === preload bank ===
    blob = data_io.load_bank_blob()
    keys = data_io.filter_keys(blob["entries"], split="train", solo_pass=True)[:args.n_preload]
    b_raw = data_io.b_stack_for_keys(blob["entries"], keys, target_norm=15.0,
                                      device=args.device, dtype=torch.float32)

    P = make_projector(d, rank=args.rank).to(args.device).float()
    if args.projector_ckpt:
        P.load_state_dict(torch.load(args.projector_ckpt, map_location=args.device))
        print(f"[e03] loaded projector from {args.projector_ckpt}")

    bank.frozen = False
    proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
    bank.slots[args.bank_layer] = proj
    bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
    bank.frozen = True

    # === config 3: bank_on (preloaded, projector at init = identity) ===
    nll_on = chunked_nll(model, ids, ctx=args.ctx, device=args.device, run_forward=fwd_lpl)

    drift_off = nll_off - nll_base
    drift_on = nll_on - nll_base
    rel_drift_on = drift_on / max(nll_base, 1e-6)

    out = {
        "model": args.model, "tokens": int(ids.numel()), "ctx": args.ctx,
        "n_preload": args.n_preload, "bank_layer": args.bank_layer,
        "projector_ckpt": args.projector_ckpt,
        "nll_base": nll_base, "ppl_base": math.exp(nll_base),
        "nll_bank_off": nll_off, "ppl_bank_off": math.exp(nll_off),
        "nll_bank_on": nll_on, "ppl_bank_on": math.exp(nll_on),
        "drift_off_vs_base": drift_off,
        "drift_on_vs_base": drift_on,
        "rel_drift_on_vs_base": rel_drift_on,
        "verdict": {
            "off_bit_equal": abs(drift_off) <= 0.001,
            "on_within_5pct": abs(rel_drift_on) <= 0.05,
            "rule": "bank_off must ≈ base; bank_on rel drift ≤ 5%",
        },
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e03] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
