"""Exp32 Φ3 — Model-attached evaluation of trained MLPGatedRouter.

Mirrors Exp31's 5-gate panel but injects at MLP output of each decoder layer
instead of inside attention. Bank = (M_K, M_V) per layer captured at
relation_last on the write prompt (K = MLP-input, V = MLP-output).

Variants:
  base                          no bank
  mlp_full_learned              learned router + softmax-mixed readout   <- MAIN
  mlp_topk1_learned             learned router + top-1 hard routing      <- MAIN-T1
  mlp_gate_off                  bank present but gate forced off (identity sanity)
  mlp_fixed_gate                gate fixed = 1 (full strength, no learning)
  mlp_minus_correct_learned     bank minus correct fact
  mlp_meanV_learned             V replaced by mean (Gate C)
  mlp_shuffled_factids_learned  K/V identity scrambled (Gate D)
  mlp_shuffled_router_learned   router trained on shuffled pairs (Gate E)  [optional]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from deltamemory.memory.mlp_gated_injector import (  # noqa: E402
    MLPGatedConfig, MLPGatedInjector, MLPGatedRouter, MLPMemoryBank,
)
from deltamemory.memory.anb_capture_sweep import derive_relation_phrase, resolve_extended_capture  # noqa: E402

from atb_validation_v1._lib import (  # noqa: E402
    load_model, seed_everything, continuation_logp, first_token_rank,
)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt  # noqa: E402


SPLITS_DIR = Path(__file__).parents[1] / "exp31_learned_k_adapter" / "data" / "splits"


def resolve_pos(site: str, row: dict, tok, prompt: str, add_special: bool = True) -> int | None:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=add_special)
    am = enc["attention_mask"][0]
    rel = derive_relation_phrase(row.get("prompt", ""))
    spec = resolve_extended_capture(
        site=site, write_prompt=prompt,
        subject=row.get("subject", ""), relation_phrase=rel,
        object_str=row.get("target_new", ""),
        tokenizer=tok, attention_mask_row=am, add_special_tokens=add_special,
    )
    if spec is None or not spec.token_positions:
        return None
    return spec.token_positions[-1]


def build_mlp_bank(model, tok, injector, rows, device) -> tuple[MLPMemoryBank, list[str]]:
    """Capture (K_in, V_out) at relation_last for each row."""
    L = injector.num_layers
    d_model = int(model.config.hidden_size)
    K_list: list[torch.Tensor] = []
    V_list: list[torch.Tensor] = []
    kept: list[str] = []
    for row in rows:
        wp = row.get("write_prompt")
        if wp is None:
            wp = build_write_prompt(row, row["target_new"])
            if wp is None:
                continue
        pos = resolve_pos("relation_last", row, tok, wp)
        if pos is None:
            continue
        enc = tok(wp, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        K, V = injector.capture_at_pos(ids, pos=int(pos), attention_mask=am)
        K_list.append(K)
        V_list.append(V)
        kept.append(str(row["id"]))
    if not kept:
        return MLPMemoryBank(M_K=[torch.empty(0, d_model) for _ in range(L)],
                             M_V=[torch.empty(0, d_model) for _ in range(L)],
                             fact_ids=[]), []
    K_stack = torch.stack(K_list, dim=0).float()   # (N, L, D)
    V_stack = torch.stack(V_list, dim=0).float()
    bank = MLPMemoryBank(
        M_K=[K_stack[:, l] for l in range(L)],
        M_V=[V_stack[:, l] for l in range(L)],
        fact_ids=kept,
    )
    return bank, kept


def bank_minus(bank: MLPMemoryBank, fid: str) -> MLPMemoryBank:
    if fid not in bank.fact_ids:
        return bank
    idx = bank.fact_ids.index(fid)
    keep = [i for i in range(bank.n_facts) if i != idx]
    new_ids = [bank.fact_ids[i] for i in keep]
    return MLPMemoryBank(
        M_K=[bank.M_K[l][keep] for l in range(bank.n_layers)],
        M_V=[bank.M_V[l][keep] for l in range(bank.n_layers)],
        fact_ids=new_ids,
    )


def bank_meanV(bank: MLPMemoryBank) -> MLPMemoryBank:
    new_V = []
    for l in range(bank.n_layers):
        m = bank.M_V[l].mean(dim=0, keepdim=True).expand_as(bank.M_V[l]).contiguous()
        new_V.append(m)
    return MLPMemoryBank(M_K=[k.clone() for k in bank.M_K], M_V=new_V, fact_ids=list(bank.fact_ids))


def bank_shuffle_factids(bank: MLPMemoryBank, rng: torch.Generator) -> MLPMemoryBank:
    perm = torch.randperm(bank.n_facts, generator=rng)
    new_V = [bank.M_V[l][perm].clone() for l in range(bank.n_layers)]
    return MLPMemoryBank(M_K=[k.clone() for k in bank.M_K], M_V=new_V, fact_ids=list(bank.fact_ids))


@torch.no_grad()
def eval_with_bank(model, tok, injector, prompt, tn, tt, device,
                   bank, router, gate_mode, topk, alpha):
    if bank is None or bank.n_facts == 0 or alpha == 0.0:
        # base path
        lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
        lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)
        return {"log_p_new": float(lp_new), "log_p_true": float(lp_true),
                "margin": float(lp_new - lp_true), "target_rank": int(rank)}

    # tokenise prompt to find the last token position to inject at
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    last_idx = int(enc["attention_mask"][0].sum().item()) - 1

    injector.cfg = MLPGatedConfig(
        key_dim=router.key_dim, temperature=0.07, topk=int(topk),
        inject_only_last_token=True, eta=float(alpha), gate_mode=gate_mode,
    )
    injector.install(bank, router, last_token_idx=last_idx)
    try:
        lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
        lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    finally:
        injector.remove()
    return {"log_p_new": float(lp_new), "log_p_true": float(lp_true),
            "margin": float(lp_new - lp_true), "target_rank": int(rank)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--router", required=True)
    ap.add_argument("--router-shuffled", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--alphas", default="0.1,0.3,1.0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()

    injector = MLPGatedInjector(model, MLPGatedConfig())
    L = injector.num_layers
    d_model = int(model.config.hidden_size)
    print(f"L={L} d_model={d_model}")

    def load_router(path: str) -> MLPGatedRouter:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        r = MLPGatedRouter(num_layers=ck["n_layers"], d_model=ck["d_model"], key_dim=ck["key_dim"])
        r.load_state_dict(ck["state_dict"])
        r = r.to(args.device).eval()
        return r

    router = load_router(args.router)
    router_shuf = load_router(args.router_shuffled) if args.router_shuffled else None

    # Load test facts + distractors
    test_facts = json.loads((SPLITS_DIR / "test.json").read_text())
    dist_ids = set(json.loads((SPLITS_DIR / "distractors.json").read_text()))
    dist_path = REPO / "experiments" / "X1_bank_scaling" / "distractors.jsonl"
    dists: list[dict] = []
    with open(dist_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("fact_id") in dist_ids:
                r["id"] = r["fact_id"]
                r.setdefault("target_true", "")
                dists.append(r)
    test_rows = []
    for r in test_facts:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        rr = dict(r); rr["write_prompt"] = wp
        test_rows.append(rr)
    eval_rows = test_rows[: args.n]
    n_dist = max(0, args.bank_size - len(test_rows))
    print(f"eval={len(eval_rows)}  bank pool={len(test_rows)} test + {min(n_dist, len(dists))} distractors")

    (out_dir / "env.json").write_text(json.dumps({
        "model": args.model, "device": args.device, "dtype": args.dtype,
        "n": args.n, "bank_size": args.bank_size, "alphas": alphas, "seeds": seeds,
        "router": args.router, "router_shuffled": args.router_shuffled,
        "site": "mlp_output", "key_dim": int(router.key_dim),
    }, indent=2))

    fout = (out_dir / "cells.jsonl").open("w")
    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)

        bank_input = (test_rows + dists)[: args.bank_size]
        bank, kept = build_mlp_bank(model, tok, injector, bank_input, args.device)
        print(f"[seed={seed}] bank built: {len(kept)}/{len(bank_input)} slots")
        kept_set = set(kept)
        mean_bank = bank_meanV(bank)
        shuf_bank = bank_shuffle_factids(bank, rng)

        n_done = 0
        for row in eval_rows:
            fid = str(row["id"])
            if fid not in kept_set:
                continue
            tn = row["target_new"]
            tt = row.get("target_true") or ""
            if not tt:
                continue
            q = render_query(row)

            m = eval_with_bank(model, tok, injector, q, tn, tt, args.device,
                               None, router, "off", 0, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid, "alpha": 0.0,
                                   "variant": "base", **m}) + "\n")

            minus = bank_minus(bank, fid)
            specs = [
                ("mlp_full_learned",            bank,      router,     "learned",   0),
                ("mlp_topk1_learned",           bank,      router,     "learned",   1),
                ("mlp_gate_off",                bank,      router,     "off",       0),
                ("mlp_fixed_gate",              bank,      router,     "fixed_one", 0),
                ("mlp_minus_correct_learned",   minus,     router,     "learned",   0),
                ("mlp_meanV_learned",           mean_bank, router,     "learned",   0),
                ("mlp_shuffled_factids_learned",shuf_bank, router,     "learned",   0),
            ]
            if router_shuf is not None:
                specs.append(("mlp_shuffled_router_learned", bank, router_shuf, "learned", 0))

            for a in alphas:
                for (vname, bnk, rtr, gmode, tk) in specs:
                    m = eval_with_bank(model, tok, injector, q, tn, tt, args.device,
                                       bnk, rtr, gmode, tk, a)
                    fout.write(json.dumps({"seed": seed, "fact_id": fid, "alpha": float(a),
                                            "variant": vname, "topk": tk, **m}) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-6)
                eta = (len(eval_rows) - n_done) / max(rate, 1e-6)
                print(f"  seed={seed} {n_done}/{len(eval_rows)} rate={rate:.2f}/s eta={eta:.0f}s", flush=True)

        del bank, minus, mean_bank, shuf_bank
        gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close()
    print(f"done in {time.time()-t0:.0f}s -> {out_dir/'cells.jsonl'}")


if __name__ == "__main__":
    main()
