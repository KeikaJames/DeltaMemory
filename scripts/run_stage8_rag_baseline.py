#!/usr/bin/env python3
"""Stage 8.5: head-to-head vs RAG baselines.

Two baselines:

  - vector_rag (closed-book, fair):  Same KeyProjector pipeline as
    Stage 8 v3 (mean-pooled address embed -> learned MLP key), retrieves
    a slot, but the bank stores the FROZEN value-token embedding rather
    than the Writer output. No Writer is trained. The retrieved frozen
    embedding is injected at the answer position exactly like the Stage 8
    pipeline. This isolates the contribution of the trained Writer.

  - text_rag (open-book ceiling):    Same retrieval, but instead of
    injecting a vector, we expand the read prompt with the retrieved
    text "<address> -> <value>" and ask the frozen base to predict the
    value token. Conceptually this is "Memorizing-Transformer-style RAG
    over (address, value) string pairs". It is allowed to see the value
    token in context — open-book by definition. We include it as an
    upper-bound reference, not as a fair head-to-head competitor.

Both are evaluated on the same fact pool and address-key projector
trained on the same InfoNCE objective, so retrieval recall is held
fixed across the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage8 import (  # noqa: E402
    FastWeightBank, KeyProjector, build_facts,
    _address_key_from_embeddings, _address_token_ids,
    _forward_read_with_injection, _tokenize_read_prompts,
)
from rcvhc.gemma.model_adapter import load_model_bundle  # noqa: E402


@torch.no_grad()
def train_key_projector_only(*, model, key_proj, addr_pooled_all,
                              n, steps, batch_size, lr, temperature, hard_negs,
                              device, seed):
    """Train ONLY the KeyProjector on InfoNCE address discrimination.
    No writer; no closed-book CE. The shared retrieval head for fair
    comparison across baselines.
    """
    import random as _random
    optim = torch.optim.AdamW(key_proj.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = _random.Random(seed + 7919)
    log_every = max(1, steps // 20)
    for step in range(1, steps + 1):
        idxs = [rng.randrange(n) for _ in range(batch_size)]
        anchor_pooled = addr_pooled_all[torch.tensor(idxs, device=device)]
        with torch.enable_grad():
            keys_anchor = key_proj(anchor_pooled)
            if hard_negs > 0:
                batch_set = set(idxs)
                extras: list[int] = []
                while len(extras) < hard_negs:
                    cand = rng.randrange(n)
                    if cand not in batch_set:
                        extras.append(cand)
                        batch_set.add(cand)
                extra_pooled = addr_pooled_all[torch.tensor(extras, device=device)]
                keys_extra = key_proj(extra_pooled)
                cand_keys = torch.cat([keys_anchor, keys_extra], dim=0)
            else:
                cand_keys = keys_anchor
            a_n = F.normalize(keys_anchor, dim=-1)
            c_n = F.normalize(cand_keys, dim=-1)
            sim = a_n @ c_n.t() / temperature
            labels = torch.arange(keys_anchor.shape[0], device=device)
            loss = F.cross_entropy(sim, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if step % log_every == 0 or step == 1:
            print(f"[stage8.5][key-proj-only] step={step}/{steps} loss={loss.item():.4f}", flush=True)


@torch.no_grad()
def evaluate_vector_rag(*, model, tokenizer, facts, key_proj, device, alpha,
                        eval_batch_size):
    n = len(facts)
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    read_prompts_all = [f.read_prompt for f in facts]
    addresses_all = [f.address for f in facts]
    input_ids_all, attn_all, last_pos_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)
    addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)

    # Build the bank: bank.v[s] = frozen embed(value_token_id of slot s).
    embed = model.get_input_embeddings()
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")
    bank = FastWeightBank(n_slots=n, hidden=hidden, key_dim=key_proj.proj[0].out_features).to(device=device, dtype=torch.float32)
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        v = embed(value_token_ids_all[slots]).float()
        k = key_proj(addr_pooled_all[slots])
        bank.write(slots, v, k)

    # Retrieve and inject.
    correct_top1 = 0
    correct_recall = 0
    total = 0
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        q = key_proj(addr_pooled_all[slots])
        pred = bank.retrieve(q)
        correct_recall += int((pred == slots).sum().item())
        inj = bank.read(pred)
        ids = input_ids_all[slots]; am = attn_all[slots]; lp = last_pos_all[slots]
        tgt = value_token_ids_all[slots]
        logits = _forward_read_with_injection(model, ids, am, lp, inj, alpha)
        pred_tok = logits.argmax(dim=-1)
        correct_top1 += int((pred_tok == tgt).sum().item())
        total += int(tgt.numel())
    return {"top1": correct_top1 / total, "recall_at_1": correct_recall / total, "n": total}


@torch.no_grad()
def evaluate_text_rag(*, model, tokenizer, facts, key_proj, device, eval_batch_size):
    """Open-book RAG ceiling. Retrieves top-1 (address, value) pair via the
    same KeyProjector and prepends 'address -> value' to the read prompt.
    """
    n = len(facts)
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    addresses_all = [f.address for f in facts]
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)
    addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)

    # Build a key-only bank to retrieve from. Values come from facts list.
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")
    bank = FastWeightBank(n_slots=n, hidden=hidden, key_dim=key_proj.proj[0].out_features).to(device=device, dtype=torch.float32)
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        k = key_proj(addr_pooled_all[slots])
        bank.write(slots, torch.zeros(end - start, hidden, device=device), k)

    # Build prompts that include the retrieved (address, value) pair.
    correct_top1 = 0
    correct_recall = 0
    total = 0
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        q = key_proj(addr_pooled_all[slots])
        pred = bank.retrieve(q)
        correct_recall += int((pred == slots).sum().item())
        # Construct retrieval-augmented prompts.
        ra_prompts = []
        for true_slot, p_slot in zip(slots.cpu().tolist(), pred.cpu().tolist()):
            f_query = facts[true_slot]
            f_retrieved = facts[p_slot]
            ra = (
                f"Memory note: slot {f_retrieved.address} has value {f_retrieved.value_token_str}.\n"
                f"Atlas slot {f_query.address}\n"
                f"Recall the payload value for this slot. The value is"
            )
            ra_prompts.append(ra)
        ids, am, lp = _tokenize_read_prompts(tokenizer, ra_prompts, device)
        tgt = value_token_ids_all[slots]
        logits = _forward_read_with_injection(model, ids, am, lp, None, alpha=0.0)
        pred_tok = logits.argmax(dim=-1)
        correct_top1 += int((pred_tok == tgt).sum().item())
        total += int(tgt.numel())
    return {"top1": correct_top1 / total, "recall_at_1": correct_recall / total, "n": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-facts", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--key-dim", type=int, default=256)
    parser.add_argument("--retrieval-temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-hard-negatives", type=int, default=0)
    parser.add_argument("--ours-summary",
                        help="Path to Stage 8 v3 summary.json for the same N/seed; we read its retrieved top1 for direct comparison.")
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[stage8.5] loading {args.model} on {args.device} ({args.dtype})", flush=True)
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

    addresses_all = [f.address for f in facts]
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)
    with torch.no_grad():
        addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)

    key_proj = KeyProjector(hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)

    t0 = time.time()
    train_key_projector_only(
        model=model, key_proj=key_proj, addr_pooled_all=addr_pooled_all,
        n=n, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        temperature=args.retrieval_temperature, hard_negs=args.retrieval_hard_negatives,
        device=device, seed=args.seed,
    )
    train_secs = time.time() - t0

    vec = evaluate_vector_rag(model=model, tokenizer=tokenizer, facts=facts,
                              key_proj=key_proj, device=device,
                              alpha=args.alpha,
                              eval_batch_size=args.eval_batch_size)
    txt = evaluate_text_rag(model=model, tokenizer=tokenizer, facts=facts,
                            key_proj=key_proj, device=device,
                            eval_batch_size=args.eval_batch_size)

    ours_top1 = None
    g2_pass = None
    if args.ours_summary:
        try:
            ours = json.loads(Path(args.ours_summary).read_text())
            ours_top1 = ours["metrics"]["bank_inject_retrieved"]["top1"]
            g2_pass = ours_top1 >= vec["top1"]
        except Exception as e:
            print(f"[stage8.5] could not load ours-summary: {e}", flush=True)

    summary = {
        "stage": "8.5_rag_head_to_head",
        "args": vars(args),
        "n_facts": n,
        "hidden_size": hidden,
        "train_seconds": train_secs,
        "vector_rag": vec,
        "text_rag_open_book_ceiling": txt,
        "ours_retrieved_top1": ours_top1,
        "gates": {
            "G2_ours_ge_vector_rag": {"value_ours": ours_top1, "value_vector_rag": vec["top1"], "pass": g2_pass},
        },
    }
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "delta_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[stage8.5] === RAG HEAD-TO-HEAD ===")
    print(f"  vector_rag (closed-book, frozen value-embed):  top1={vec['top1']:.3f}  recall@1={vec['recall_at_1']:.3f}")
    print(f"  text_rag   (open-book ceiling, RA prompt):     top1={txt['top1']:.3f}  recall@1={txt['recall_at_1']:.3f}")
    if ours_top1 is not None:
        print(f"  ours       (Stage 8 v3 trained Writer):        top1={ours_top1:.3f}")
        print(f"  G2 ours >= vector_rag: {'PASS' if g2_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
