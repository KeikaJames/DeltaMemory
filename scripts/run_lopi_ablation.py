"""R-2/R-3 Phase R smoke: Dynamic LOPI ablation grid on GPT-2.

Implements the A0..A4 ablation grid from `reports/cleanroom/lopi_v33/PREREGISTRATION.md §4`
inside the same hooked-attention forward used by `scripts/run_mHC3_bank_injection.py`.
The goal is *not* the full 630-cell sweep yet — that's R-3 — but to establish
the LOPI-enabled hook plumbing, prove A0 ≡ legacy bit-equal, and produce one
matched-α numeric snapshot for every variant.

Usage
-----
    python scripts/run_lopi_ablation.py \
        --model gpt2 --device mps --dtype bfloat16 \
        --alpha 1.0 --seed 0 \
        --out reports/cleanroom/lopi_v33_smoke/results.json

Output per cell records: variant, alpha, arch, lift, drift, plus the LOPI
config dict for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse helpers from the mHC3 sweep — single source of truth for bank IO.
from scripts.run_mHC3_bank_injection import (
    NEUTRAL_PROMPTS,
    FALSE_FACTS,
    _write_fact,
    _get_attention,
    seq_nll,
    load_architecture,
)
from deltamemory.memory.lopi import LOPIConfig, LOPIState, apply_lopi


VARIANTS = {
    "A0": dict(enabled=False),
    "A1": dict(enabled=True, orthogonal=True, gaussian=False, derivative=False),
    "A2": dict(enabled=True, orthogonal=True, gaussian=True, derivative=False),
    "A3": dict(enabled=True, orthogonal=True, gaussian=True, derivative=True),
    "A4": dict(enabled=True, orthogonal=False, gaussian=True, derivative=True),
}


@torch.no_grad()
def _read_with_bank_lopi(model, tok, read_prompt, bank, alpha,
                         n_layers, device, *,
                         lopi_cfg: LOPIConfig,
                         lopi_state: LOPIState,
                         return_full_logits: bool = False):
    """LOPI-aware bank-injected forward.

    Mirrors `scripts.run_mHC3_bank_injection._read_with_bank` but splits
    out_orig / out_bank explicitly so `apply_lopi()` can wrap the bank
    contribution per the v3.3 PREREGISTRATION.
    """
    enc = tok(read_prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)

    handles = []

    def make_hook(layer_idx):
        entry = bank[layer_idx]
        mk = entry["K"].to(device)
        mv = entry["V"].to(device)
        n_embd = mk.size(-1)

        def hook(module, input, output):
            # Bit-equal short-circuit: at α=0 the bank contribution is
            # mathematically zero regardless of LOPI components, so we
            # bypass the manual attention reimpl (which doesn't match
            # GPT-2's native head-mask/bias handling exactly) and return
            # the model's native output.  Preserves H5 red-line.
            if alpha == 0.0:
                return output

            x = input[0]
            B, T, _ = x.shape
            qkv = module.c_attn(x)
            split_dim = module.split_size
            q, k, v = qkv.split(split_dim, dim=-1)
            n_head = module.num_heads
            head_dim = n_embd // n_head

            q_r = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k_r = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v_r = v.view(B, T, n_head, head_dim).transpose(1, 2)
            mk_r = mk.view(1, n_head, 1, head_dim).expand(B, -1, 1, -1)
            mv_r = mv.view(1, n_head, 1, head_dim).expand(B, -1, 1, -1)

            k_cat = torch.cat([k_r, mk_r], dim=-2)             # (B, H, T+1, D)
            scale = head_dim ** -0.5
            scores = torch.matmul(q_r, k_cat.transpose(-2, -1)) * scale
            # Apply causal mask over the first T columns; bank slot (last) is
            # always visible. Match seq_nll_with_bank's mask in mHC3 fix.
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                              diagonal=1)
            scores[..., :T].masked_fill_(mask, float("-inf"))

            w = torch.softmax(scores.float(), dim=-1).to(q_r.dtype)

            # Split readouts so LOPI can wrap the bank contribution.
            out_orig = torch.matmul(w[..., :T], v_r)
            out_bank = torch.matmul(w[..., T:], alpha * mv_r)

            out_bank = apply_lopi(
                out_bank_native=out_bank,
                v_ctx_readout=out_orig,
                q_post=q_r,
                layer_idx=layer_idx,
                state=lopi_state,
                cfg=lopi_cfg,
            )

            # Update t-1 residual norm cache (mean L2 over heads/positions).
            lopi_state.prev_residual_norms[layer_idx] = float(
                torch.linalg.vector_norm(out_orig.float(), ord=2, dim=-1).mean().item()
            )

            attn_out = (out_orig + out_bank).transpose(1, 2).contiguous().view(B, T, n_embd)
            attn_out = module.c_proj(attn_out)
            attn_out = module.resid_dropout(attn_out)
            return (attn_out, None)
        return hook

    for li in range(n_layers):
        attn = _get_attention(model, li)
        h = attn.register_forward_hook(make_hook(li))
        handles.append(h)

    out = model(input_ids=ids, attention_mask=am, use_cache=True)
    last_pos = am.sum(dim=1).item() - 1
    for h in handles:
        h.remove()

    if return_full_logits:
        return out.logits[0].float()
    return out.logits[0, last_pos].float()


def seq_nll_with_lopi(model, tok, prompt, bank, alpha, n_layers, device,
                      lopi_cfg, lopi_state) -> float:
    logits = _read_with_bank_lopi(
        model, tok, prompt, bank, alpha, n_layers, device,
        lopi_cfg=lopi_cfg, lopi_state=lopi_state,
        return_full_logits=True,
    )
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].to(device)
    targets = ids[1:]
    logp = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
    return -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item()


def _load_model(model_id: str, device: str, dtype_str: str):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    tok = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def run_cell(model, tok, n_layers: int, device: str,
             facts, neutrals, base_nlls,
             alpha: float, variant_id: str, cfg_kwargs: dict, seed: int) -> dict:
    """Run a single (arch × scale × seed × alpha × variant) cell."""
    torch.manual_seed(seed)
    cfg = LOPIConfig(**cfg_kwargs)
    state = LOPIState(num_layers=n_layers)

    lifts = []
    for f in facts:
        wp = f["write"]
        rp = f["read"]
        target_id = tok(" " + f["target"], add_special_tokens=False)["input_ids"][0]
        bank = _write_fact(model, tok, wp, n_layers, device)
        state.reset()
        with torch.no_grad():
            base_logits = model(**tok(rp, return_tensors="pt").to(device)).logits[0, -1].float()
        base_lp = torch.nn.functional.log_softmax(base_logits, dim=-1)[target_id].item()
        inj_logits = _read_with_bank_lopi(
            model, tok, rp, bank, alpha, n_layers, device,
            lopi_cfg=cfg, lopi_state=state,
        )
        inj_lp = torch.nn.functional.log_softmax(inj_logits, dim=-1)[target_id].item()
        lifts.append(inj_lp - base_lp)

    neutral_bank = _write_fact(
        model, tok,
        "Fact: The Sun is a star at the centre of the Solar System.",
        n_layers, device,
    )
    state.reset()
    inj_nlls = []
    for p in neutrals:
        inj_nlls.append(seq_nll_with_lopi(
            model, tok, p, neutral_bank, alpha, n_layers, device, cfg, state,
        ))
        state.reset()

    drift = sum(i - b for b, i in zip(base_nlls, inj_nlls)) / len(neutrals)
    return {
        "variant": variant_id,
        "config": cfg.asdict(),
        "alpha": alpha,
        "seed": seed,
        "n_layers": n_layers,
        "mean_lift": sum(lifts) / len(lifts),
        "lifts": lifts,
        "mean_drift": drift,
        "inj_nlls": inj_nlls,
        "base_nlls": base_nlls,
        "drift_metric": "seq_nll_diff_neutral",
        "n_facts": len(facts),
        "n_neutral": len(neutrals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-facts", type=int, default=2)
    ap.add_argument("--n-neutral", type=int, default=2)
    # Full-sweep mode (R-3):
    ap.add_argument("--sweep", action="store_true",
                    help="Run full R-3 ablation grid: archs × scales × seeds × alphas × variants.")
    ap.add_argument("--archs", nargs="+", default=["residual", "hc", "mhc"])
    ap.add_argument("--scales", nargs="+", default=["gpt2", "gpt2-medium"])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+",
                    default=["A0", "A1", "A2", "A3", "A4"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if not args.sweep:
        # Legacy single-cell smoke mode (R-2).
        torch.manual_seed(args.seed)
        print(f"[lopi-smoke] loading {args.model} on {args.device} ({args.dtype})")
        model, tok = _load_model(args.model, args.device, args.dtype)
        n_layers = model.config.n_layer
        print(f"[lopi-smoke] n_layers={n_layers}")
        facts = FALSE_FACTS[: args.n_facts]
        neutrals = NEUTRAL_PROMPTS[: args.n_neutral]
        base_nlls = [seq_nll(model, tok, p, args.device) for p in neutrals]
        cells = []
        for vid in args.variants:
            cell = run_cell(model, tok, n_layers, args.device,
                            facts, neutrals, base_nlls,
                            args.alpha, vid, VARIANTS[vid], args.seed)
            cell["arch"] = "residual"
            cell["model"] = args.model
            cells.append(cell)
            print(f"[{vid}] lift={cell['mean_lift']:+.4f}  drift={cell['mean_drift']:+.4f}")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(cells, indent=2))
        print(f"[lopi-smoke] wrote {out_path}")
        return

    # ---- Full R-3 sweep ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    facts = FALSE_FACTS  # full 5
    neutrals = NEUTRAL_PROMPTS  # full 5
    total_cells = (len(args.archs) * len(args.scales) * len(args.seeds) *
                   len(args.alphas) * len(args.variants))
    print(f"[lopi-sweep] total cells = {total_cells}")
    print(f"[lopi-sweep] archs={args.archs} scales={args.scales} "
          f"seeds={args.seeds} alphas={args.alphas} variants={args.variants}")
    all_cells = []
    cell_idx = 0
    for scale in args.scales:
        for arch in args.archs:
            print(f"[lopi-sweep] loading arch={arch} scale={scale}", flush=True)
            tok, model = load_architecture(arch, scale, args.device, dtype)
            n_layers = model.config.n_layer
            base_nlls = [seq_nll(model, tok, p, args.device) for p in neutrals]
            for seed in args.seeds:
                for alpha in args.alphas:
                    for vid in args.variants:
                        cell_idx += 1
                        cell = run_cell(model, tok, n_layers, args.device,
                                        facts, neutrals, base_nlls,
                                        alpha, vid, VARIANTS[vid], seed)
                        cell["arch"] = arch
                        cell["scale"] = scale
                        cell["model"] = scale
                        all_cells.append(cell)
                        print(f"[{cell_idx:>4}/{total_cells}] "
                              f"{scale}/{arch} seed={seed} α={alpha} {vid} "
                              f"lift={cell['mean_lift']:+.4f} drift={cell['mean_drift']:+.4f}",
                              flush=True)
                        # Incremental write every cell so a crash mid-sweep
                        # doesn't lose all results.
                        out_path.write_text(json.dumps(all_cells, indent=2))
            del model, tok
            if args.device == "mps":
                torch.mps.empty_cache()
    print(f"[lopi-sweep] DONE: wrote {len(all_cells)} cells to {out_path}")


if __name__ == "__main__":
    main()
