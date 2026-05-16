"""e10 top-K retrieval — does content-sensitive retrieval beat capacity?

Research question
-----------------
e11 wave-3 showed that the trained rank-64 K-projector + ANY non-empty bank
(real, random Gaussian, single replicated row, constant vector) produces
roughly the same NLL drop. The "bank" might just be a content-agnostic
capacity boost from the trained projection, not associative memory.

This driver tests a falsifier: if top-K retrieval over a REAL bank gives a
CLEARLY LARGER NLL improvement than (a) top-K over a RANDOM bank or
(b) all-attend over a RANDOM bank, then content-sensitive retrieval is doing
real work. If they tie, capacity wins.

Variants (--variant)
--------------------
    topk_cosine_real_K1     : real bank, top-1 cosine
    topk_cosine_real_K8     : real bank, top-8 cosine
    topk_cosine_real_K64    : real bank, top-64 cosine (matches projector rank)
    topk_cosine_random_K8   : random L2=15 bank, top-8 cosine
    topk_random_indices_K8  : real bank, K random rows (no cosine)
    all_attend_real         : real bank, all N entries (canonical e01)
    all_attend_random_renorm15 : random L2=15 bank, all N entries (matches e11/n1)

Pass criterion (cross-variant; verifiable from JSON output by analysis script):
    Δ_real(topk_cosine_real_K8) <= -1.0
    AND post_real(all_attend_random_renorm15) >= post_real(topk_cosine_real_K8) + 1.0

A single-run JSON only records its own Δ_real; the cross-variant comparison
must be done after running both `topk_cosine_real_K8` and
`all_attend_random_renorm15`.

Query default: ``mean_embed`` — query = mean of the input token embeddings
(no extra forward pass). ``last_hidden`` does an extra no-bank forward to grab
hidden_states[bank_layer] at the last token. Documented in the output JSON.

Output JSON: ``e10_<variant>_seed<seed>.json``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)
from v2.core.retrieval import topk_cosine


# ---------------------------------------------------------------------------
# Variant config
VARIANTS = [
    "topk_cosine_real_K1",
    "topk_cosine_real_K8",
    "topk_cosine_real_K64",
    "topk_cosine_random_K8",
    "topk_random_indices_K8",
    "all_attend_real",
    "all_attend_random_renorm15",
]


def variant_config(variant: str, default_K: int) -> dict:
    """Returns: {use_random_bank, retrieval_mode, K}.
    retrieval_mode in {"cosine", "random_indices", "all"}.
    K only meaningful for cosine / random_indices.
    """
    if variant == "topk_cosine_real_K1":
        return dict(use_random_bank=False, retrieval_mode="cosine", K=1)
    if variant == "topk_cosine_real_K8":
        return dict(use_random_bank=False, retrieval_mode="cosine", K=8)
    if variant == "topk_cosine_real_K64":
        return dict(use_random_bank=False, retrieval_mode="cosine", K=64)
    if variant == "topk_cosine_random_K8":
        return dict(use_random_bank=True, retrieval_mode="cosine", K=8)
    if variant == "topk_random_indices_K8":
        return dict(use_random_bank=False, retrieval_mode="random_indices", K=8)
    if variant == "all_attend_real":
        return dict(use_random_bank=False, retrieval_mode="all", K=default_K)
    if variant == "all_attend_random_renorm15":
        return dict(use_random_bank=True, retrieval_mode="all", K=default_K)
    raise ValueError(f"unknown variant {variant}")


# ---------------------------------------------------------------------------
# Forward helpers (mirror e01)
def forward_lpl_k2(model, bank, heads, enc, *, grad=False):
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True,
                      force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
              use_cache=False)
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True,
                      force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


# ---------------------------------------------------------------------------
# Query computation
def compute_query(model, enc, *, mode: str, bank_layer: int) -> torch.Tensor:
    """Return a [d] query vector for retrieval."""
    if mode == "mean_embed":
        with torch.no_grad():
            emb = model.get_input_embeddings()(enc.input_ids)  # [B, T, d]
        return emb[0].float().mean(dim=0)
    elif mode == "last_hidden":
        # No-bank forward to grab hidden_states[bank_layer] at last token.
        prev_state = getattr(model, "lpl_state", None)
        model.lpl_state = None
        with torch.no_grad():
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        use_cache=False, output_hidden_states=True)
        model.lpl_state = prev_state
        # hidden_states[bank_layer] = input to layer bank_layer (after embed = idx 0)
        h = out.hidden_states[bank_layer]  # [B, T, d]
        return h[0, -1].float()
    else:
        raise ValueError(f"unknown query_mode {mode}")


# ---------------------------------------------------------------------------
# Bank-population helper. Mutates `bank.slots[bank_layer]` in place.
def populate_bank(*, bank, bank_layer, b_pool_full, P, retrieval_mode, K,
                  query, rng_torch, grad_track: bool):
    """Select rows from b_pool_full per retrieval_mode, project, write to bank.

    b_pool_full : [N, d] float on device — full preload bank (real or random).
    P           : projector (residual_apply convention).
    retrieval_mode: "cosine" | "random_indices" | "all".
    query       : [d] float (only used by cosine).
    rng_torch   : torch.Generator for random_indices reproducibility.
    grad_track  : if True, do not torch.no_grad the projector pass (training).
    """
    N = b_pool_full.shape[0]
    if retrieval_mode == "cosine":
        idx = topk_cosine(query.to(b_pool_full.device), b_pool_full, K)  # [K]
        subset = b_pool_full[idx]
    elif retrieval_mode == "random_indices":
        perm = torch.randperm(N, generator=rng_torch)[:K]
        subset = b_pool_full[perm.to(b_pool_full.device)]
    elif retrieval_mode == "all":
        subset = b_pool_full
    else:
        raise ValueError(retrieval_mode)

    if grad_track:
        proj = residual_apply(P, subset).to(dtype=torch.bfloat16)
    else:
        with torch.no_grad():
            proj = residual_apply(P, subset).to(dtype=torch.bfloat16)
    bank.frozen = False
    bank.slots[bank_layer] = proj
    bank.tags[bank_layer] = [(0, -1)] * proj.shape[0]
    bank.frozen = True


def make_random_bank(shape, *, target_norm: float, device, dtype, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = torch.randn(*shape, generator=g)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-9) * target_norm
    return n.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Split builder — mirrors e01 canonical path
def build_split(entries, *, n_train, n_test, n_preload, seed):
    keys = list(entries.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng.shuffle(train_keys); rng.shuffle(test_keys)
    train_keys = train_keys[:n_train]
    test_keys = test_keys[:n_test]
    preload_pool = data_io.filter_keys(entries, split="train", solo_pass=True)
    preload_pool = [k for k in preload_pool if k not in set(train_keys)]
    preload_keys = preload_pool[:n_preload]
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    return train_items, test_items, preload_keys


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=8,
                   help="K for retrieval (variant may override).")
    p.add_argument("--query_mode", choices=["mean_embed", "last_hidden"],
                   default="mean_embed")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else \
        HERE / f"e10_{args.variant}_seed{args.seed}.json"

    cfg = variant_config(args.variant, default_K=args.K)
    use_random_bank = cfg["use_random_bank"]
    retrieval_mode = cfg["retrieval_mode"]
    K = cfg["K"]

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    train_items, test_items, preload_keys = build_split(
        entries, n_train=args.n_train, n_test=args.n_eval,
        n_preload=args.n_preload, seed=args.seed,
    )
    print(f"[e10:{args.variant}] train={len(train_items)} test={len(test_items)} "
          f"preload={len(preload_keys)} seed={args.seed} K={K} "
          f"retrieval={retrieval_mode} random_bank={use_random_bank} "
          f"query_mode={args.query_mode}")

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    # Build the FULL bank pool we retrieve over.
    b_real = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                      device=args.device, dtype=torch.float32)
    if use_random_bank:
        b_pool_full = make_random_bank(b_real.shape, target_norm=15.0,
                                       device=args.device, dtype=torch.float32,
                                       seed=args.seed + 1000)
        print(f"[e10:{args.variant}] using RANDOM bank pool shape={tuple(b_pool_full.shape)} L2=15")
    else:
        b_pool_full = b_real
        print(f"[e10:{args.variant}] using REAL bank pool shape={tuple(b_pool_full.shape)} L2=15")

    # Independent random pool for the "rand" eval (matches e01's rand_bank).
    b_rand_eval = make_random_bank(b_pool_full.shape, target_norm=15.0,
                                   device=args.device, dtype=torch.float32,
                                   seed=args.seed + 7777)

    P = make_projector(d, rank=args.rank).to(args.device).float()

    rng_torch = torch.Generator(device="cpu").manual_seed(args.seed)

    # -----------------------------------------------------------------------
    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    def eval_lpl(items, *, override_pool=None, zero_bank=False, bank_off=False):
        """Evaluate with retrieval rule.

        override_pool: replace b_pool_full for this eval (used for 'rand' eval).
        zero_bank   : write zeros into the selected subset.
        bank_off    : empty bank entirely (variant 'off').
        """
        pool = override_pool if override_pool is not None else b_pool_full
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            if bank_off:
                bank.frozen = False
                bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device,
                                                          dtype=torch.bfloat16)
                bank.tags[args.bank_layer] = []
                bank.frozen = True
            else:
                q = compute_query(model, enc, mode=args.query_mode,
                                  bank_layer=args.bank_layer) if retrieval_mode == "cosine" \
                    else torch.zeros(d, device=args.device)
                populate_bank(bank=bank, bank_layer=args.bank_layer,
                              b_pool_full=pool, P=P,
                              retrieval_mode=retrieval_mode, K=K, query=q,
                              rng_torch=rng_torch, grad_track=False)
                if zero_bank:
                    bank.frozen = False
                    bank.slots[args.bank_layer] = torch.zeros_like(
                        bank.slots[args.bank_layer]
                    )
                    bank.frozen = True
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    # -----------------------------------------------------------------------
    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    pre_rand = eval_lpl(test_items, override_pool=b_rand_eval)
    pre_zero = eval_lpl(test_items, zero_bank=True)
    pre_off = eval_lpl(test_items, bank_off=True)
    print(f"[e10:{args.variant}] BEFORE: base={base:.4f}  real={pre_real:.4f}  "
          f"rand={pre_rand:.4f}  zero={pre_zero:.4f}  off={pre_off:.4f}")

    # === training ===
    rng = random.Random(args.seed)
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)

        if retrieval_mode == "cosine":
            q = compute_query(model, enc, mode=args.query_mode,
                              bank_layer=args.bank_layer)
        else:
            q = torch.zeros(d, device=args.device)
        populate_bank(bank=bank, bank_layer=args.bank_layer,
                      b_pool_full=b_pool_full, P=P,
                      retrieval_mode=retrieval_mode, K=K, query=q,
                      rng_torch=rng_torch, grad_track=True)

        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 25 == 0 or step == 0:
            recent = sum(losses[-25:]) / min(25, len(losses))
            print(f"  [e10:{args.variant}] step {step+1}/{args.steps} "
                  f"loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    print(f"[e10:{args.variant}] training done in {time.time()-t0:.1f}s")

    # === post-train eval ===
    post_real = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, override_pool=b_rand_eval)
    post_zero = eval_lpl(test_items, zero_bank=True)
    post_off = eval_lpl(test_items, bank_off=True)
    print(f"[e10:{args.variant}] AFTER:  base={base:.4f}  real={post_real:.4f}  "
          f"rand={post_rand:.4f}  zero={post_zero:.4f}  off={post_off:.4f}")

    delta_real = post_real - base
    # Per-variant single-run verdict; the cross-variant capacity check needs
    # both `topk_cosine_real_K8` and `all_attend_random_renorm15` JSONs.
    if args.variant == "topk_cosine_real_K8":
        verdict = {
            "pass": delta_real <= -1.0,
            "rule": "Δ_real <= -1.0 (single-run; full pass also requires "
                    "post_real(all_attend_random_renorm15) >= "
                    "post_real(topk_cosine_real_K8) + 1.0)",
            "delta_signed": delta_real,
        }
    else:
        verdict = {
            "pass": None,
            "rule": "informational; see cross-variant comparison",
            "delta_signed": delta_real,
        }

    out = {
        "variant": args.variant,
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "K": K,
        "retrieval_mode": retrieval_mode,
        "use_random_bank": use_random_bank,
        "query_mode": args.query_mode,
        "before": {"base": base, "real": pre_real, "rand": pre_rand,
                   "zero": pre_zero, "off": pre_off},
        "after": {"base": base, "real": post_real, "rand": post_rand,
                  "zero": post_zero, "off": post_off},
        "delta_real": delta_real,
        "verdict": verdict,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "n_train_params": sum(p.numel() for p in trainable),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e10:{args.variant}] -> {out_path}  verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
