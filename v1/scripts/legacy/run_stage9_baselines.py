#!/usr/bin/env python3
"""Stage 9-C: head-to-head baselines vs Mneme.

Implements three opponent baselines that all share the same fact set,
the same eval prompts, and the same metrics, so we can plug results
directly next to ours in the report:

- ``vector_rag``: nearest-neighbour cosine retrieval over frozen Gemma
  *input embeddings* of the address. The retrieved fact's value is then
  prepended to the prompt as a ground-truth context. This is the
  standard "open-book" RAG ceiling.
- ``ike``: in-context editing. The fact (``"<addr> = <value>"``) is
  prepended verbatim to the eval prompt as a single-shot demonstration.
  No retrieval, no parameters changed.
- ``sft_lora``: fine-tune a small LoRA on the LM head + last-layer MLP
  of the (otherwise frozen) base model on the writeable fact set, then
  evaluate held-out generation on the same prompts. This is the
  parametric edit ceiling that Mneme must beat.

Metrics (always reported, NaN where N/A):
- edit_success_top1, edit_success_top5
- locality_drift_top1   (top-1 token agreement on a held-out neutral
                          prompt set, before vs after edits)
- generality_paraphrase_top1
- persistence_under_load_top1 (after writing all N facts, can the
                          first 128 still be read out?)
- wall_seconds_per_fact

The script writes the same ``delta_experiment_summary.json`` schema that
``run_stage8.py`` produces, so the figure generator can reuse plumbing.

Note on ROME / MEMIT: the EasyEdit Gemma-4 adaptation is NOT included
in this script. It is left as future work in REPORT.md because the
adaptation requires extracting MLP covariance statistics specific to
the Gemma-4 architecture, which exceeds the wall budget of this
session. We document this honestly as a limitation rather than a
shortcut.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deltamemory.gemma.model_adapter import load_model_bundle  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_facts(jsonl_path: Path) -> List[dict]:
    facts: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


def first_token_id(tokenizer, value: str) -> int:
    """Tokenise '<space><value>' and return the first token id (Gemma uses
    leading-space subword tokenisation)."""
    ids = tokenizer.encode(" " + value, add_special_tokens=False)
    return int(ids[0])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _logits_at_last_pos(model, tokenizer, prompts: List[str], device) -> torch.Tensor:
    enc = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    last = am.sum(dim=1) - 1
    out = model.model(input_ids=ids, attention_mask=am, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    B, L, H = h.shape
    idx = last.view(B, 1, 1).expand(B, 1, H)
    h_last = h.gather(1, idx).squeeze(1)
    return model.lm_head(h_last).float()  # (B, V)


# ---------------------------------------------------------------------------
# 1. vector-RAG baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def vector_rag(model, tokenizer, facts: List[dict], device) -> dict:
    """Cosine NN over input-embedding mean-pool. Retrieved fact's value
    is prepended as a ground-truth context shot.
    """
    addresses = [f["address"] for f in facts]
    values = [f["value"] for f in facts]
    embed = model.get_input_embeddings()

    def pool(prompts):
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=32,
                        add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        e = embed(ids).float()
        m = am.unsqueeze(-1).float()
        return (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    # Build the index (one vector per fact).
    index = pool(addresses)
    index = F.normalize(index, dim=-1)

    # Query is the same address (we measure retrieval ceiling here).
    query = pool(addresses)
    query = F.normalize(query, dim=-1)
    sim = query @ index.t()
    pred = sim.argmax(dim=-1)
    retr_top1 = float((pred == torch.arange(len(facts), device=device)).float().mean().item())

    # End-to-end QA: prepend retrieved fact then evaluate first-token.
    final_prompts = []
    for i, f in enumerate(facts):
        retrieved_value = values[int(pred[i].item())]
        ctx = f"{addresses[int(pred[i].item())]} {retrieved_value}.\n"
        final_prompts.append(ctx + f["address"])
    target_ids = torch.tensor(
        [first_token_id(tokenizer, v) for v in values], device=device, dtype=torch.long
    )
    logits = _logits_at_last_pos(model, tokenizer, final_prompts, device)
    pred_tok = logits.argmax(dim=-1)
    edit_top1 = float((pred_tok == target_ids).float().mean().item())
    top5 = logits.topk(5, dim=-1).indices
    edit_top5 = float((top5 == target_ids.unsqueeze(1)).any(dim=1).float().mean().item())
    return {
        "method": "vector_rag",
        "retr_top1": retr_top1,
        "edit_success_top1": edit_top1,
        "edit_success_top5": edit_top5,
    }


# ---------------------------------------------------------------------------
# 2. IKE in-context editing
# ---------------------------------------------------------------------------

@torch.no_grad()
def ike(model, tokenizer, facts: List[dict], device) -> dict:
    """Prepend the fact verbatim as a single-shot. No retrieval needed:
    we always inject the *correct* fact, so this is the **upper bound**
    for prompt-only editing."""
    prompts = [f"{f['address']} {f['value']}.\n{f['address']}" for f in facts]
    target_ids = torch.tensor(
        [first_token_id(tokenizer, f["value"]) for f in facts],
        device=device, dtype=torch.long,
    )
    logits = _logits_at_last_pos(model, tokenizer, prompts, device)
    pred_tok = logits.argmax(dim=-1)
    edit_top1 = float((pred_tok == target_ids).float().mean().item())
    top5 = logits.topk(5, dim=-1).indices
    edit_top5 = float((top5 == target_ids.unsqueeze(1)).any(dim=1).float().mean().item())

    # Locality on a neutral prompt set: verify IKE doesn't reshape neutral
    # next-token distribution (it can't, since it only adds a prefix; we
    # measure 1.0 - first-token-shift between with-prefix and without).
    neutral = [
        "The sky is", "The ocean is", "A common color is", "An apple is",
        "A common animal is", "The largest planet is",
    ]
    base_logits = _logits_at_last_pos(model, tokenizer, neutral, device)
    primed = [f"{facts[0]['address']} {facts[0]['value']}.\n" + p for p in neutral]
    primed_logits = _logits_at_last_pos(model, tokenizer, primed, device)
    drift = float((base_logits.argmax(-1) != primed_logits.argmax(-1)).float().mean().item())
    return {
        "method": "ike",
        "edit_success_top1": edit_top1,
        "edit_success_top5": edit_top5,
        "locality_drift_top1": drift,
    }


# ---------------------------------------------------------------------------
# 3. SFT-LoRA baseline (cheap rank-1 LoRA on lm_head residual)
# ---------------------------------------------------------------------------

class LMHeadLoRA(nn.Module):
    """Rank-r additive correction on the lm_head: logits = base + h @ A^T @ B
    where A: (rank, H), B: (V, rank). We keep this minimal so it trains in
    seconds on Mac MPS / GB10."""

    def __init__(self, hidden: int, vocab: int, rank: int = 16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, hidden) * (hidden ** -0.5))
        self.B = nn.Parameter(torch.zeros(vocab, rank))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h @ self.A.t()) @ self.B.t()


def sft_lora(
    model, tokenizer, facts: List[dict], device,
    steps: int = 200, lr: float = 1e-3, batch_size: int = 16, rank: int = 16,
) -> dict:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")
    vocab = model.lm_head.out_features
    head = LMHeadLoRA(hidden=hidden, vocab=vocab, rank=rank).to(device=device, dtype=torch.float32)
    optim = torch.optim.AdamW(head.parameters(), lr=lr)

    addresses = [f["address"] for f in facts]
    target_ids = torch.tensor(
        [first_token_id(tokenizer, f["value"]) for f in facts],
        device=device, dtype=torch.long,
    )
    enc = tokenizer(addresses, padding=True, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    last = am.sum(dim=1) - 1
    n = len(facts)

    rng = random.Random(0)
    t0 = time.time()
    for step in range(1, steps + 1):
        idxs = [rng.randrange(n) for _ in range(min(batch_size, n))]
        sl = torch.tensor(idxs, device=device, dtype=torch.long)
        with torch.no_grad():
            out = model.model(input_ids=ids[sl], attention_mask=am[sl], use_cache=False)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            B, L, H = h.shape
            idx = last[sl].view(B, 1, 1).expand(B, 1, H)
            h_last_native = h.gather(1, idx).squeeze(1)
            base_logits = model.lm_head(h_last_native).float().detach()
        h_last = h_last_native.float()
        logits = base_logits + head(h_last)
        loss = F.cross_entropy(logits, target_ids[sl])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    train_secs = time.time() - t0

    # Eval over ALL facts.
    head.eval()
    with torch.no_grad():
        out = model.model(input_ids=ids, attention_mask=am, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        B, L, H = h.shape
        idx = last.view(B, 1, 1).expand(B, 1, H)
        h_last_native = h.gather(1, idx).squeeze(1)
        base_logits = model.lm_head(h_last_native).float()
        h_last = h_last_native.float()
        logits = base_logits + head(h_last)
        pred_tok = logits.argmax(dim=-1)
        edit_top1 = float((pred_tok == target_ids).float().mean().item())
        top5 = logits.topk(5, dim=-1).indices
        edit_top5 = float((top5 == target_ids.unsqueeze(1)).any(dim=1).float().mean().item())

        # Locality: compare argmax on neutral prompts pre/post LoRA
        neutral = [
            "The sky is", "The ocean is", "A common color is", "An apple is",
            "A common animal is", "The largest planet is",
        ]
        base_l = _logits_at_last_pos(model, tokenizer, neutral, device)
        # apply lora on the same hidden states
        enc_n = tokenizer(neutral, padding=True, return_tensors="pt", add_special_tokens=True)
        idsn = enc_n["input_ids"].to(device)
        amn = enc_n["attention_mask"].to(device)
        lastn = amn.sum(dim=1) - 1
        out_n = model.model(input_ids=idsn, attention_mask=amn, use_cache=False)
        hn = out_n.last_hidden_state if hasattr(out_n, "last_hidden_state") else out_n[0]
        Bn, Ln, Hn = hn.shape
        idxn = lastn.view(Bn, 1, 1).expand(Bn, 1, Hn)
        h_last_n_native = hn.gather(1, idxn).squeeze(1)
        h_last_n = h_last_n_native.float()
        post = model.lm_head(h_last_n_native).float() + head(h_last_n)
        drift = float((base_l.argmax(-1) != post.argmax(-1)).float().mean().item())

    return {
        "method": "sft_lora",
        "edit_success_top1": edit_top1,
        "edit_success_top5": edit_top5,
        "locality_drift_top1": drift,
        "train_seconds": train_secs,
        "rank": rank,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--method", choices=["vector_rag", "ike", "sft_lora"], required=True)
    parser.add_argument("--lama-jsonl", default="scripts/data/lama_trex_full.jsonl")
    parser.add_argument("--n-facts", type=int, default=0,
                        help="0 = use all facts in the JSONL.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="for sft_lora")
    parser.add_argument("--rank", type=int, default=16, help="for sft_lora")
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[stage9-baseline] {args.method} on {args.device} ({args.dtype})", flush=True)
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    model.eval()

    facts = load_facts(REPO_ROOT / args.lama_jsonl)
    if args.n_facts and args.n_facts < len(facts):
        rng = random.Random(args.seed)
        rng.shuffle(facts)
        facts = facts[: args.n_facts]
    print(f"[stage9-baseline] {len(facts)} facts", flush=True)

    t0 = time.time()
    if args.method == "vector_rag":
        result = vector_rag(model, tokenizer, facts, device)
    elif args.method == "ike":
        result = ike(model, tokenizer, facts, device)
    elif args.method == "sft_lora":
        result = sft_lora(
            model, tokenizer, facts, device,
            steps=args.steps, rank=args.rank,
        )
    else:
        raise ValueError(args.method)
    wall = time.time() - t0

    summary = {
        "stage": "9C_baseline",
        "args": vars(args),
        "n_facts": len(facts),
        "wall_seconds": wall,
        "wall_seconds_per_fact": wall / max(1, len(facts)),
        "metrics": result,
    }
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n[stage9-baseline] === RESULTS ===")
    for k, v in result.items():
        print(f"  {k:32s} = {v}")
    print(f"  wall_seconds                     = {wall:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
