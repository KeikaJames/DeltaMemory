"""Exp33 — Re-attention readout via AttnNativeBank, sourced from CounterFact.

Phase B of the post-Exp32 plan. Tests whether the native joint-softmax
sparse-attention bank — `Attn(Q, [K_seq; M_K], [V_seq; M_V])` — produces
Gate B signal when:

  * Bank size N = 200 (125 test facts + 75 distractors)
  * Bank K/V are captured by the model's *own* attention forward pass on
    the write prompt (the "AttnNativeBank.write_fact" path, NOT projected
    from Exp32 residual-stream captures).
  * Read prompts are the test paraphrases.
  * α sweep ∈ {0.05, 0.1, 0.3}; top-k ∈ {1, full}.

This is the same architecture that Exp27 (`EXP27_SPARSE_VERDICT.md`)
already falsified at N=200 with different fact splits. We re-run on the
Exp31/32 splits as a clean control, because:

  - the LS diagnostic showed routing is fine (76% honest test top-1);
  - Exp32 with α-additive readout gave Gate B = 0/375;
  - Exp27 with joint-softmax readout gave Gate B failure at N=200 too.

A positive Gate B here would invalidate that pattern.  A negative result
locks the verdict: bank-style external memory cannot inject fact identity
into Qwen3-4B regardless of readout protocol (additive OR joint softmax).

We measure only the four gates that matter:

  Gate B   margin(target_new) − margin(target_true) > 0
  Gate A   topk − minus_correct  (bank contribution direction)
  Gate D   topk − shuffled_factids  (fact identity bound to K)
  retr     argmax-bank-attention-weight = correct fact

at the last token of each paraphrase, frozen base model.
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
from deltamemory import AttnNativeBank, AttnNativePatcher, fresh_bank, write_fact  # noqa: E402


SPLITS = Path(__file__).resolve().parents[1] / "exp31_learned_k_adapter" / "data" / "splits"


def first_target_id(tokenizer, target: str) -> int:
    """Tokenize ' target' and take the first id (leading-space convention)."""
    ids = tokenizer(" " + target.strip(), add_special_tokens=False).input_ids
    return int(ids[0])


def build_write_prompt(row: dict) -> str:
    body = row["prompt"].format(row["subject"]).strip()
    return f"{body} {row['target_new']}."


@torch.no_grad()
def margins_at_lastpos(model, tokenizer, prompt: str, t_new_id: int, t_true_id: int):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=am, use_cache=False)
    last = int(am.sum(dim=1).item() - 1)
    logp = F.log_softmax(out.logits[0, last].float(), dim=-1)
    return float(logp[t_new_id].item()), float(logp[t_true_id].item())


@torch.no_grad()
def eval_one(model, tokenizer, patcher, bank, row, alpha, fact_idx,
             variant: str, perm: list[int] | None = None):
    """Return margin = logp(t_new) − logp(t_true), averaged over paraphrases."""
    t_new = first_target_id(tokenizer, row["target_new"])
    t_true = first_target_id(tokenizer, row["target_true"])
    margins = []
    paraphrases = row.get("paraphrase_prompts", [])[:2]

    # Build a temporary mask: which bank entries are "active" for this variant?
    # We can't trivially remove a fact, but we *can* simulate variants by
    # rearranging the M_K/M_V tensors and restoring.
    saved_MK = [t.clone() for t in bank.M_K]
    saved_MV = [t.clone() for t in bank.M_V]
    try:
        if variant == "minus_correct":
            for l in range(bank.num_layers):
                keep = [i for i in range(bank.M_K[l].shape[0]) if i != fact_idx]
                bank.M_K[l] = saved_MK[l][keep]
                bank.M_V[l] = saved_MV[l][keep]
        elif variant == "shuffled_factids":
            assert perm is not None
            for l in range(bank.num_layers):
                bank.M_V[l] = saved_MV[l][perm]

        for p in paraphrases:
            with patcher.patched(), patcher.injecting(bank, alpha=alpha):
                m_new, m_true = margins_at_lastpos(model, tokenizer, p, t_new, t_true)
            margins.append(m_new - m_true)
    finally:
        for l in range(bank.num_layers):
            bank.M_K[l] = saved_MK[l]
            bank.M_V[l] = saved_MV[l]
    return sum(margins) / max(1, len(margins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-test", type=int, default=125)
    ap.add_argument("--n-distractors", type=int, default=75)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "run_qwen_exp33"))
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()

    test = json.load(open(SPLITS / "test.json"))[: args.n_test]
    # Use train facts as distractors (full fact dicts, not just ids)
    distractors_pool = json.load(open(SPLITS / "train.json"))
    distractors = distractors_pool[: args.n_distractors]

    bank_facts = test + distractors
    print(f"bank size N = {len(bank_facts)}  (test={len(test)} distractors={len(distractors)})",
          flush=True)

    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)

    t0 = time.time()
    for i, row in enumerate(bank_facts):
        wp = build_write_prompt(row)
        write_fact(patcher, bank, tok,
                   write_prompt=wp, fact_id=row["id"], address=row["subject"])
        if (i + 1) % 25 == 0:
            print(f"  write {i+1}/{len(bank_facts)}  ({time.time()-t0:.0f}s)", flush=True)

    # Permutation for shuffled_factids (over the full bank)
    g = torch.Generator().manual_seed(1000 + args.seed)
    perm = torch.randperm(len(bank_facts), generator=g).tolist()

    rows = []
    for alpha in args.alphas:
        for fact_idx, row in enumerate(test):
            for variant in ("topk_full", "minus_correct", "shuffled_factids"):
                # base is α=0 (recovered separately below; no need per-α)
                m = eval_one(model, tok, patcher, bank, row, alpha, fact_idx,
                             variant, perm=perm if variant == "shuffled_factids" else None)
                rows.append({"alpha": alpha, "fact_idx": fact_idx, "id": row["id"],
                             "variant": variant, "margin": m})
        print(f"alpha={alpha} done at {time.time()-t0:.0f}s", flush=True)

    # baseline: α=0 (sparse-attention with bank present but read mass=0)
    for fact_idx, row in enumerate(test):
        m = eval_one(model, tok, patcher, bank, row, 0.0, fact_idx, "topk_full")
        rows.append({"alpha": 0.0, "fact_idx": fact_idx, "id": row["id"],
                     "variant": "base", "margin": m})

    out_jsonl = out_dir / "cells.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_jsonl}  ({len(rows)} rows)")

    # Aggregate
    def agg(filt):
        ms = [r["margin"] for r in rows if filt(r)]
        return (sum(ms) / len(ms), len(ms)) if ms else (float("nan"), 0)

    base_mean, _ = agg(lambda r: r["variant"] == "base")
    summary = {"meta": {"N_bank": len(bank_facts), "n_test": len(test),
                        "seed": args.seed, "alphas": args.alphas},
               "base_mean": base_mean, "by_alpha": {}}
    for a in args.alphas:
        for v in ("topk_full", "minus_correct", "shuffled_factids"):
            m, n = agg(lambda r, a=a, v=v: r["alpha"] == a and r["variant"] == v)
            summary["by_alpha"].setdefault(str(a), {})[v] = {"mean": m, "n": n}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
