"""Exp35 Φ0 — Build the Fact-LoRA bank.

For each fact in train+val+test, run minimal ROME at L=5 to obtain a
rank-1 factorisation (b, a) of the down_proj update Δ. Save to bank.pt.

Anti-cheat:
  - C6 norm budget: monitor ‖Δ_i‖_F; flag/drop outliers above
    `outlier_mult × median`.
  - Solo gate: only keep facts whose solo-patch margin > 0 on the
    canonical read prompt AND on >=1 of 2 paraphrases.
  - C9 bit-equal: assert weights restored after each fact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

SPLITS = Path(__file__).resolve().parents[1] / "exp31_learned_k_adapter" / "data" / "splits"
HERE = Path(__file__).resolve().parent


# --- shared helpers (mirror exp34/run_exp34.py) ---

def first_target_id(tokenizer, target: str) -> int:
    ids = tokenizer(" " + target.strip(), add_special_tokens=False).input_ids
    return int(ids[0])


def subject_last_pos(tokenizer, prompt: str, subject: str) -> int:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    for variant in (" " + subject.strip(), subject.strip()):
        sids = tokenizer(variant, add_special_tokens=False).input_ids
        if not sids:
            continue
        for i in range(len(ids) - len(sids), -1, -1):
            if ids[i:i + len(sids)] == sids:
                return i + len(sids) - 1
    return len(ids) - 1


def capture_k_star(model, tokenizer, prompt: str, edit_layer: int, pos: int):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    captured = {}

    def hook(module, inp, out):
        captured["x"] = inp[0][0, pos].detach().clone()

    h = model.model.layers[edit_layer].mlp.down_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        h.remove()
    return captured["x"]


def compute_v_star(model, tokenizer, read_prompt: str, edit_layer: int,
                   t_new_id: int, t_true_id: int, n_steps: int, lr: float):
    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    delta = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    layer = model.model.layers[edit_layer]

    enc = tokenizer(read_prompt, return_tensors="pt", add_special_tokens=True).to(device)
    last_pos = int(enc["attention_mask"][0].sum().item() - 1)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            h_out = out[0]
            h_out[0, last_pos] = h_out[0, last_pos] + delta.to(h_out.dtype)
            return (h_out,) + out[1:]
        out[0, last_pos] = out[0, last_pos] + delta.to(out.dtype)
        return out

    h = layer.register_forward_hook(hook)
    try:
        for _ in range(n_steps):
            opt.zero_grad()
            out = model(**enc, use_cache=False)
            logp = F.log_softmax(out.logits[0, last_pos].float(), dim=-1)
            loss = -(logp[t_new_id] - 0.5 * logp[t_true_id]) + 1e-3 * (delta ** 2).sum()
            loss.backward()
            opt.step()
    finally:
        h.remove()
    return delta.detach()


def rome_update_factors(W, k_star, v_star, lam: float):
    """Return (b, a) with b a^T = (v* - W k*) k*^T / (||k*||^2 + lam).

    Δ is rank-1 by construction, so set b = (v* - W k*) and
    a = k* / (||k*||^2 + lam).
    """
    k = k_star.to(W.dtype)
    v = v_star.to(W.dtype)
    Wk = W @ k
    denom = float((k.float() @ k.float()).item()) + lam
    a = k / denom
    b = v - Wk
    return b, a


def apply_factors(model, edit_layer, factors):
    """factors: list of (b, a). Returns saved W_old."""
    W = model.model.layers[edit_layer].mlp.down_proj.weight
    W_old = W.data.clone()
    if factors:
        # Σ b_i a_i^T = B A^T with B = stack(b), A = stack(a)
        B = torch.stack([b.to(W.dtype) for b, _ in factors], dim=1)   # (d_out, k)
        A = torch.stack([a.to(W.dtype) for _, a in factors], dim=1)   # (d_in, k)
        W.data = W_old + B @ A.t()
    return W_old


def restore(model, edit_layer, W_old):
    model.model.layers[edit_layer].mlp.down_proj.weight.data = W_old


@torch.no_grad()
def margin_at_last(model, tokenizer, prompt: str, t_new: int, t_true: int) -> float:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, use_cache=False)
    last = int(enc["attention_mask"][0].sum().item() - 1)
    logp = F.log_softmax(out.logits[0, last].float(), dim=-1)
    return float(logp[t_new].item() - logp[t_true].item())


def assert_bit_equal(model, edit_layer, W_ref):
    cur = model.model.layers[edit_layer].mlp.down_proj.weight.data
    if not torch.equal(cur, W_ref):
        raise RuntimeError("C9 bit-equal violated: W not restored to base.")


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--v-steps", type=int, default=25)
    ap.add_argument("--v-lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--n-test", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "bank.pt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train = json.load(open(SPLITS / "train.json"))
    val = json.load(open(SPLITS / "val.json"))
    test = json.load(open(SPLITS / "test.json"))[: args.n_test]
    all_facts = [("train", i, r) for i, r in enumerate(train)] \
                + [("val", i, r) for i, r in enumerate(val)] \
                + [("test", i, r) for i, r in enumerate(test)]
    print(f"bank size = {len(all_facts)} (train={len(train)} val={len(val)} test={len(test)})", flush=True)

    W = model.model.layers[args.edit_layer].mlp.down_proj.weight
    W_ref = W.data.clone()

    entries = {}
    norms = []
    t0 = time.time()

    for idx, (split, sidx, row) in enumerate(all_facts):
        fid = row["id"]
        t_new = first_target_id(tok, row["target_new"])
        t_true = first_target_id(tok, row["target_true"])

        # Write prompt = read prompt #0 (canonical). subject token at last
        # subject-token position serves as both anchor and reading site.
        canon = row["prompt"].format(row["subject"])
        pos = subject_last_pos(tok, canon, row["subject"])

        # k*
        k_star = capture_k_star(model, tok, canon, args.edit_layer, pos)
        # v* (optimised on canonical)
        v_star = compute_v_star(model, tok, canon, args.edit_layer,
                                t_new, t_true, args.v_steps, args.v_lr)

        # rank-1 factors
        b, a = rome_update_factors(W, k_star, v_star, args.lam)
        delta_norm = float((b.float().norm() * a.float().norm()).item())
        norms.append(delta_norm)

        # solo gate
        W_old = apply_factors(model, args.edit_layer, [(b, a)])
        try:
            paraphrases = list(row.get("paraphrase_prompts", []))[:2]
            read_prompts = [canon] + paraphrases
            margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
            canonical_pass = margins[0] > 0
            para_pass = sum(1 for m in margins[1:] if m > 0)
            solo_pass = bool(canonical_pass and para_pass >= 1)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)

        entries[fid] = {
            "split": split,
            "split_idx": sidx,
            "fact_idx_global": idx,
            "subject": row["subject"],
            "target_true": row["target_true"],
            "target_new": row["target_new"],
            "b": b.cpu().to(torch.float32),
            "a": a.cpu().to(torch.float32),
            "delta_norm": delta_norm,
            "solo_margins": [float(m) for m in margins],
            "solo_pass": solo_pass,
        }

        if (idx + 1) % 25 == 0:
            recent_pass = sum(1 for fid_, e in list(entries.items())[-25:] if e["solo_pass"]) / 25
            print(f"  {idx+1}/{len(all_facts)} ({time.time()-t0:.0f}s)  "
                  f"recent_solo_pass={recent_pass:.0%}  "
                  f"||Δ||_F mean={sum(norms[-25:])/25:.2f}", flush=True)

    # outlier filter (C6)
    norms_t = torch.tensor(norms)
    med = float(norms_t.median())
    cutoff = 3.0 * med
    flagged = [fid for fid, e in entries.items() if e["delta_norm"] > cutoff]
    for fid in flagged:
        entries[fid]["norm_outlier"] = True

    summary = {
        "meta": {"edit_layer": args.edit_layer, "v_steps": args.v_steps,
                 "v_lr": args.v_lr, "lam": args.lam, "seed": args.seed,
                 "n_total": len(all_facts), "n_test": args.n_test},
        "solo_pass_rate": sum(1 for e in entries.values() if e["solo_pass"]) / len(entries),
        "solo_pass_by_split": {
            s: {
                "n": sum(1 for e in entries.values() if e["split"] == s),
                "pass": sum(1 for e in entries.values()
                            if e["split"] == s and e["solo_pass"]),
            }
            for s in ("train", "val", "test")
        },
        "delta_norm_median": med,
        "delta_norm_max": float(norms_t.max()),
        "delta_norm_outlier_cutoff": cutoff,
        "n_norm_outliers": len(flagged),
    }
    torch.save({"entries": entries, "summary": summary}, args.out)
    json.dump(summary, open(HERE / "run_qwen_exp35" / "phi0_summary.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
