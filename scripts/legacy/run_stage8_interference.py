#!/usr/bin/env python3
"""Stage 8.3: sequential-write interference / retention curve.

Protocol:
  1. Train Writer + KeyProjector jointly on the full pool of N facts as in
     ``run_stage8.py`` (same config). This gives us the trained components
     that know how to encode value tokens and address keys.
  2. Freeze Writer + KeyProjector.
  3. Sequential write phase: starting with an empty FastWeightBank, write
     facts one by one in slot order 0..N-1. After every checkpoint c
     (e.g. c in {N/8, N/4, N/2, 3N/4, N}), measure retrieved-top1 over
     the *already-written* slots [0..c). Plot retention(c).

The diagnostic question: as the bank fills up, do early slots stay
recoverable? If yes, the system has anti-interference. If retrieval
recall on early slots collapses as N grows in the bank, the
KeyProjector's keys for early slots are getting drowned by later
near-duplicates.

Gate G3: retention over the earliest N/8 slots after writing all N >= 0.80.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse all primitives from run_stage8.py
from scripts.legacy.run_stage8 import (  # noqa: E402
    FastWeightBank, KeyProjector, Writer, build_facts, build_facts_lama,
    _address_key_from_embeddings, _address_token_ids,
    _forward_read_with_injection, _tokenize_read_prompts, _value_embeds,
    train,
)
from deltamemory.gemma.model_adapter import load_model_bundle  # noqa: E402


@torch.no_grad()
def evaluate_subset(*, model, bank, key_proj, facts, slot_subset,
                    input_ids_all, attn_all, last_pos_all,
                    addr_pooled_all, value_token_ids_all,
                    device, alpha, eval_batch_size):
    """Closed-book retrieved top1 over a subset of slots that have been
    written into the bank.
    """
    n_written = bank.n_written  # type: ignore[attr-defined]
    if n_written == 0 or slot_subset.numel() == 0:
        return {"top1": 0.0, "recall_at_1": 0.0, "n": 0}
    correct_top1 = 0
    correct_recall = 0
    total = 0
    n = len(facts)
    for start in range(0, slot_subset.numel(), eval_batch_size):
        end = min(start + eval_batch_size, slot_subset.numel())
        slots = slot_subset[start:end].to(device)
        ids = input_ids_all[slots]
        am = attn_all[slots]
        lp = last_pos_all[slots]
        tgt = value_token_ids_all[slots]
        # Restrict retrieval to written part of the bank only.
        q = key_proj(addr_pooled_all[slots])
        q_n = F.normalize(q.float(), dim=-1)
        k_n = F.normalize(bank.k[:n_written].float(), dim=-1)
        sims = q_n @ k_n.t()
        pred = sims.argmax(dim=-1)
        correct_recall += int((pred == slots).sum().item())
        inj = bank.read(pred)
        logits = _forward_read_with_injection(model, ids, am, lp, inj, alpha)
        pred_tok = logits.argmax(dim=-1)
        correct_top1 += int((pred_tok == tgt).sum().item())
        total += int(tgt.numel())
    return {"top1": correct_top1 / max(1, total),
            "recall_at_1": correct_recall / max(1, total),
            "n": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-facts", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--key-dim", type=int, default=256)
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-hard-negatives", type=int, default=0)
    parser.add_argument("--checkpoints", type=str, default="0.125,0.25,0.5,0.75,1.0",
                        help="Comma-separated fractions of N at which to measure retention.")
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[stage8.3] loading {args.model} on {args.device} ({args.dtype})", flush=True)
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.eval()
    device = bundle.device
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")

    facts = build_facts(tokenizer, n_facts=args.n_facts, seed=args.seed)
    n = len(facts)

    # Patch in n_written tracking for FastWeightBank used by retention sweep.
    bank = FastWeightBank(n_slots=n, hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)
    bank.n_written = 0  # type: ignore[attr-defined]
    writer = Writer(hidden=hidden).to(device=device, dtype=torch.float32)
    key_proj = KeyProjector(hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)

    # Train writer + key_proj jointly on the full pool. This step still uses
    # in-batch closed-book CE supervision against the *full* bank (rebuilt
    # each step from current writer state), exactly like run_stage8.py.
    t0 = time.time()
    train(
        model=model, tokenizer=tokenizer, bank=bank, writer=writer,
        key_proj=key_proj, facts=facts, device=device,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, grad_clip=args.grad_clip, seed=args.seed,
        retrieval_loss_weight=args.retrieval_loss_weight,
        retrieval_temperature=args.retrieval_temperature,
        retrieval_hard_negatives=args.retrieval_hard_negatives,
    )
    train_secs = time.time() - t0

    # Pre-compute static per-fact tensors for the sequential write phase.
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    read_prompts_all = [f.read_prompt for f in facts]
    addresses_all = [f.address for f in facts]
    input_ids_all, attn_all, last_pos_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)
    with torch.no_grad():
        addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)

    # Reset bank to empty for the sequential write phase.
    bank.v.zero_()
    bank.k.zero_()
    bank.n_written = 0  # type: ignore[attr-defined]

    # Sequential write: write slots 0..N-1 one batch at a time. Measure
    # retention at each checkpoint.
    fractions = sorted({float(x) for x in args.checkpoints.split(",")})
    cps = [max(1, int(round(f * n))) for f in fractions]
    cps = sorted(set(cps))
    history = []
    write_batch = 64

    written_so_far = 0
    next_cp_idx = 0
    while written_so_far < n:
        end = min(written_so_far + write_batch, n)
        slots = torch.arange(written_so_far, end, device=device, dtype=torch.long)
        with torch.no_grad():
            ve = _value_embeds(model, value_token_ids_all[slots]).float()
            v = writer(ve)
            k = key_proj(addr_pooled_all[slots])
        bank.write(slots, v, k)
        written_so_far = end
        bank.n_written = written_so_far  # type: ignore[attr-defined]
        # Flush all checkpoints at or below current size.
        while next_cp_idx < len(cps) and cps[next_cp_idx] <= written_so_far:
            cp = cps[next_cp_idx]
            # Retention over the EARLIEST N/8 slots, restricted to those
            # already written (so this is well-defined for small cp).
            earliest = torch.arange(min(max(1, n // 8), cp), device=device, dtype=torch.long)
            full_written = torch.arange(cp, device=device, dtype=torch.long)
            ret_early = evaluate_subset(
                model=model, bank=bank, key_proj=key_proj, facts=facts,
                slot_subset=earliest, input_ids_all=input_ids_all,
                attn_all=attn_all, last_pos_all=last_pos_all,
                addr_pooled_all=addr_pooled_all,
                value_token_ids_all=value_token_ids_all, device=device,
                alpha=args.alpha, eval_batch_size=args.eval_batch_size)
            ret_all_written = evaluate_subset(
                model=model, bank=bank, key_proj=key_proj, facts=facts,
                slot_subset=full_written, input_ids_all=input_ids_all,
                attn_all=attn_all, last_pos_all=last_pos_all,
                addr_pooled_all=addr_pooled_all,
                value_token_ids_all=value_token_ids_all, device=device,
                alpha=args.alpha, eval_batch_size=args.eval_batch_size)
            history.append({
                "checkpoint_slots_written": cp,
                "earliest_subset_size": int(earliest.numel()),
                "earliest_top1": ret_early["top1"],
                "earliest_recall_at_1": ret_early["recall_at_1"],
                "all_written_top1": ret_all_written["top1"],
                "all_written_recall_at_1": ret_all_written["recall_at_1"],
            })
            print(f"[stage8.3] cp={cp}/{n} earliest top1={ret_early['top1']:.3f} all-written top1={ret_all_written['top1']:.3f}", flush=True)
            next_cp_idx += 1

    g3_pass = history[-1]["earliest_top1"] >= 0.80 if history else False

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "8.3_interference_retention",
        "args": vars(args),
        "n_facts": n,
        "hidden_size": hidden,
        "train_seconds": train_secs,
        "history": history,
        "gates": {
            "G3_earliest_retention_top1_ge_0.80": {
                "value": history[-1]["earliest_top1"] if history else 0.0,
                "pass": g3_pass,
            }
        },
    }
    (report_dir / "delta_experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stage8.3] G3 retention(earliest) at full = {history[-1]['earliest_top1']:.3f} -> {'PASS' if g3_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
