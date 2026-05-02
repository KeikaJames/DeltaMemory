#!/usr/bin/env python3
"""Stage 8 closed-book memory test.

Tests whether a persistent fast-weight bank can store (address -> answer)
associations and retrieve them at read time WITHOUT the value token appearing
in the prompt. This is the strict closed-book gate that distinguishes "real
memory" from "in-context binding via fancy router".

Mechanism (write/read protocol):
    Bank: ``nn.Parameter(N, H)`` zero-initialised, one slot per fact.
    Writer: small MLP, hidden_state(value_token) -> bank slot vector.
    Read:  prompt = "<addr> the value is" (NO value token); forward through
           frozen Gemma; at the answer position add ``alpha * bank[slot]`` to
           the hidden state; project through lm_head; CE on answer token.

Training: end-to-end via read-phase CE. Bank is recomputed each step from
writer(value_embed). At eval, bank is populated once via writer, then
read-only forward proceeds with no value tokens anywhere.

Pass/fail gate G1: top1 >= 0.80 on the held-out (or in-distribution) read set.

Example:
    python3 scripts/run_stage8.py --n-facts 128 --steps 1500 --device cuda \\
        --dtype bfloat16 --seed 0 --report-dir reports/experiments/stage8_1_n128_seed0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rcvhc.gemma.model_adapter import load_model_bundle  # noqa: E402

# Curated single-token color vocabulary (validated under Gemma tokenizer with
# leading space). 32 entries; we draw from this for the value side.
COLOR_TOKENS: list[str] = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
    "black", "white", "gray", "silver", "gold", "violet", "indigo", "scarlet",
    "crimson", "maroon", "navy", "teal", "olive", "lime", "cyan", "magenta",
    "amber", "ivory", "ebony", "cream", "azure", "beige", "mauve", "tan",
]


# ---------------------------------------------------------------------------
# Data: synthetic single-token closed-book memory facts
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    slot: int
    address: str
    value_token_str: str   # "red"
    value_token_id: int    # tokenized id under leading space

    @property
    def write_prompt(self) -> str:
        # Visible value, only used to derive the writer input via tokenizer
        # if needed. The bank input is actually the value-token embedding,
        # not text — but we also offer this for diagnostic uses.
        return (
            f"Atlas slot {self.address}\n"
            f"PAYLOAD value: {self.value_token_str}\n"
            f"END."
        )

    @property
    def read_prompt(self) -> str:
        # Closed book: VALUE TOKEN MUST NOT APPEAR in this prompt anywhere.
        return (
            f"Atlas slot {self.address}\n"
            f"Recall the payload value for this slot. The value is"
        )


def _validate_single_token(tokenizer, word: str) -> int | None:
    """Return token id if ``word`` (with leading space) tokenizes to exactly
    one non-special token; else None."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def build_facts(tokenizer, n_facts: int, seed: int) -> list[Fact]:
    rng = random.Random(seed)
    valid: list[tuple[str, int]] = []
    for c in COLOR_TOKENS:
        tid = _validate_single_token(tokenizer, c)
        if tid is not None:
            valid.append((c, tid))
    if not valid:
        raise RuntimeError("no single-token colors found under tokenizer")
    facts: list[Fact] = []
    for slot in range(n_facts):
        # Distinct addresses; they don't have to be tokenizer-friendly,
        # they just need to differ across facts.
        address = f"S-{slot:05d}-{rng.randint(0, 9999):04d}"
        word, tid = rng.choice(valid)
        facts.append(Fact(slot=slot, address=address, value_token_str=word, value_token_id=tid))
    return facts


def build_facts_lama(tokenizer, jsonl_path: Path, seed: int, n_facts: int | None = None) -> list[Fact]:
    """Load curated LAMA-style triples from JSONL.

    Each line: {"address": "<question template>", "value": "<answer word>"}.
    Validates that ``value`` tokenizes to a single token under the model's
    tokenizer (with a leading space). The slot id is the row index.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rng.shuffle(rows)
    if n_facts is not None:
        rows = rows[:n_facts]
    facts: list[Fact] = []
    skipped = 0
    for slot, row in enumerate(rows):
        addr = row["address"]
        word = row["value"].strip()
        tid = _validate_single_token(tokenizer, word)
        if tid is None:
            skipped += 1
            continue
        facts.append(Fact(slot=slot, address=addr, value_token_str=word, value_token_id=tid))
    if skipped:
        print(f"[stage8] WARN: skipped {skipped}/{len(rows)} LAMA rows (multi-token answer under tokenizer)", flush=True)
    # Re-number slot ids contiguously.
    facts = [Fact(slot=i, address=f.address, value_token_str=f.value_token_str, value_token_id=f.value_token_id)
             for i, f in enumerate(facts)]
    return facts


# ---------------------------------------------------------------------------
# Bank + Writer
# ---------------------------------------------------------------------------

class Writer(nn.Module):
    """Maps the (frozen) value-token embedding to a bank slot vector.

    Tiny 2-layer MLP with LayerNorm + zero-initialised output projection so
    initial bank entries are the zero vector and ``no_memory`` baseline is
    well-defined.
    """

    def __init__(self, hidden: int, expansion: int = 4):
        super().__init__()
        self.up = nn.Linear(hidden, expansion * hidden, bias=True)
        self.down = nn.Linear(expansion * hidden, hidden, bias=True)
        self.ln = nn.LayerNorm(hidden)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, value_embed: torch.Tensor) -> torch.Tensor:
        h = self.up(value_embed)
        h = F.gelu(h)
        h = self.down(h)
        return self.ln(h)


class FastWeightBank(nn.Module):
    """Per-slot persistent injection vectors + per-slot address keys.

    Storage:
      - v: (n_slots, hidden) injection vectors
      - k: (n_slots, key_dim) address keys (populated at write time from
        a frozen KeyEncoder over address tokens)

    Retrieval: cosine similarity between a query key and bank.k -> top1 slot.
    """

    def __init__(self, n_slots: int, hidden: int, key_dim: int):
        super().__init__()
        self.n_slots = n_slots
        self.hidden = hidden
        self.key_dim = key_dim
        self.register_buffer("v", torch.zeros(n_slots, hidden))
        self.register_buffer("k", torch.zeros(n_slots, key_dim))

    def write(self, slots: torch.Tensor, vectors: torch.Tensor, keys: torch.Tensor | None = None) -> None:
        self.v[slots] = vectors.detach().to(self.v.dtype)
        if keys is not None:
            self.k[slots] = keys.detach().to(self.k.dtype)

    def read(self, slots: torch.Tensor) -> torch.Tensor:
        return self.v[slots]

    def retrieve(self, query_keys: torch.Tensor) -> torch.Tensor:
        """Return top1 slot id for each query (B, key_dim)."""
        q = F.normalize(query_keys.float(), dim=-1)
        k = F.normalize(self.k.float(), dim=-1)
        sims = q @ k.t()  # (B, n_slots)
        return sims.argmax(dim=-1)


def _address_token_ids(tokenizer, addresses: list[str], device, max_len: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize address strings (no special tokens) and return ids+mask."""
    enc = tokenizer(
        addresses, padding=True, truncation=True, max_length=max_len,
        add_special_tokens=False, return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _address_key_from_embeddings(model, addr_ids: torch.Tensor, addr_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the (frozen) input-token embeddings over address tokens.

    This produces a deterministic content-derived key (no learning), so a
    learned KeyProjector has to project it into a discriminative space at
    scale.
    """
    embed = model.get_input_embeddings()
    e = embed(addr_ids)  # (B, T, H)
    m = addr_mask.unsqueeze(-1).float()
    pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return pooled.float()  # (B, H)


class KeyProjector(nn.Module):
    """Maps mean-pooled address embedding -> trainable key space."""
    def __init__(self, hidden: int, key_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Forward: closed-book read with bank injection
# ---------------------------------------------------------------------------

def _tokenize_read_prompts(tokenizer, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1) - 1   # index of the last real token
    return input_ids, attention_mask, last_pos


def _value_embeds(model, value_token_ids: torch.Tensor) -> torch.Tensor:
    embed = model.get_input_embeddings()
    return embed(value_token_ids)  # (B, H)


def _forward_read_with_injection(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    last_pos: torch.Tensor,
    injection_vectors: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    """Forward through frozen base, inject at last position, project via lm_head.

    Returns logits at the last real position only: shape (B, V).
    """
    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
    )
    last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    B, L, H = last_hidden.shape
    idx = last_pos.view(B, 1, 1).expand(B, 1, H)
    h_at_answer = last_hidden.gather(1, idx).squeeze(1)  # (B, H)
    if injection_vectors is not None:
        h_at_answer = h_at_answer + alpha * injection_vectors.to(h_at_answer.dtype)
    logits = model.lm_head(h_at_answer)  # (B, V)
    # Gemma may apply final logit softcapping inside model.forward; we don't
    # replicate it here because the relative ranking on answer-token CE is
    # invariant to monotone scaling, and softcapping is monotone for finite
    # logits.
    return logits


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

@dataclass
class StepStats:
    step: int
    loss: float
    top1: float


def train(
    *,
    model,
    tokenizer,
    bank: FastWeightBank,
    writer: Writer,
    key_proj: KeyProjector,
    facts: list[Fact],
    device,
    steps: int,
    batch_size: int,
    lr: float,
    alpha: float,
    grad_clip: float,
    seed: int,
    retrieval_loss_weight: float,
    retrieval_temperature: float,
    retrieval_hard_negatives: int = 0,
) -> list[StepStats]:
    params = list(writer.parameters()) + list(key_proj.parameters())
    optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    n = len(facts)
    rng = random.Random(seed + 7919)
    history: list[StepStats] = []
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    read_prompts_all = [f.read_prompt for f in facts]
    addresses_all = [f.address for f in facts]
    input_ids_all, attn_all, last_pos_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)
    # Pre-cache mean-pooled address embeddings (frozen base, no grad). This
    # lets us cheaply pull global negatives without rerunning the embedding
    # table for every step.
    with torch.no_grad():
        addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)

    log_every = max(1, steps // 20)
    for step in range(1, steps + 1):
        idxs = [rng.randrange(n) for _ in range(batch_size)]
        slots = torch.tensor(idxs, device=device, dtype=torch.long)
        value_embeds = _value_embeds(model, value_token_ids_all[slots]).detach().float()
        bank_vectors = writer(value_embeds)
        ids = input_ids_all[slots]
        am = attn_all[slots]
        lp = last_pos_all[slots]
        targets = value_token_ids_all[slots]
        logits = _forward_read_with_injection(model, ids, am, lp, bank_vectors, alpha)
        ce = F.cross_entropy(logits.float(), targets)

        # InfoNCE retrieval. Anchors = batch slots; candidates = batch slots
        # (in-batch negatives) + optional K extra global random slots
        # (extra random negatives shared across the batch).
        anchor_pooled = addr_pooled_all[slots]
        keys_anchor = key_proj(anchor_pooled)
        if retrieval_hard_negatives > 0:
            # Sample K extra slots not in the current batch.
            batch_set = set(idxs)
            extras: list[int] = []
            while len(extras) < retrieval_hard_negatives:
                cand = rng.randrange(n)
                if cand not in batch_set:
                    extras.append(cand)
                    batch_set.add(cand)
            extra_slots = torch.tensor(extras, device=device, dtype=torch.long)
            extra_pooled = addr_pooled_all[extra_slots]
            keys_extra = key_proj(extra_pooled)
            cand_keys = torch.cat([keys_anchor, keys_extra], dim=0)
        else:
            cand_keys = keys_anchor
        a_n = F.normalize(keys_anchor, dim=-1)
        c_n = F.normalize(cand_keys, dim=-1)
        sim = a_n @ c_n.t() / retrieval_temperature  # (B, B+K)
        labels = torch.arange(keys_anchor.shape[0], device=device)
        retr_loss = F.cross_entropy(sim, labels)
        loss = ce + retrieval_loss_weight * retr_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optim.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                top1 = (logits.argmax(dim=-1) == targets).float().mean().item()
            history.append(StepStats(step=step, loss=float(loss.item()), top1=top1))
            print(f"[stage8] step={step}/{steps} ce={ce.item():.4f} retr={retr_loss.item():.4f} top1={top1:.3f}", flush=True)
    return history


@torch.no_grad()
def evaluate(
    *,
    model,
    tokenizer,
    bank: FastWeightBank,
    writer: Writer,
    key_proj: KeyProjector,
    facts: list[Fact],
    device,
    alpha: float,
    eval_batch_size: int,
) -> dict:
    n = len(facts)
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    read_prompts_all = [f.read_prompt for f in facts]
    addresses_all = [f.address for f in facts]
    input_ids_all, attn_all, last_pos_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)

    # Phase A: WRITE — populate bank.v from writer, bank.k from key_proj.
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        value_embeds = _value_embeds(model, value_token_ids_all[slots]).float()
        v = writer(value_embeds)
        addr_pooled = _address_key_from_embeddings(model, addr_ids_all[slots], addr_mask_all[slots])
        k = key_proj(addr_pooled)
        bank.write(slots, v, k)

    # Pre-compute predicted slots from address-content retrieval.
    pred_slots_all = torch.empty(n, device=device, dtype=torch.long)
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        addr_pooled = _address_key_from_embeddings(model, addr_ids_all[slots], addr_mask_all[slots])
        q = key_proj(addr_pooled)
        pred_slots_all[start:end] = bank.retrieve(q)
    retrieval_recall_at_1 = (pred_slots_all == torch.arange(n, device=device)).float().mean().item()

    def run(channel: str) -> dict:
        all_top1 = 0
        all_top10 = 0
        all_nll = 0.0
        all_count = 0
        all_paired_flip = 0
        all_paired_count = 0
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            slots = torch.arange(start, end, device=device, dtype=torch.long)
            ids = input_ids_all[slots]
            am = attn_all[slots]
            lp = last_pos_all[slots]
            tgt = value_token_ids_all[slots]
            if channel == "bank_inject_oracle":
                inj = bank.read(slots)
            elif channel == "bank_inject_retrieved":
                # Use slots predicted by address-content retrieval.
                inj = bank.read(pred_slots_all[start:end])
            elif channel == "no_memory":
                inj = None
            elif channel == "swap_paired":
                neighbor = (slots + 1) % n
                inj = bank.read(neighbor)
            else:
                raise ValueError(channel)
            logits = _forward_read_with_injection(model, ids, am, lp, inj, alpha)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tgt_lp = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            all_nll += float((-tgt_lp).sum().item())
            all_count += int(tgt.numel())
            top10 = log_probs.topk(10, dim=-1).indices
            top1 = log_probs.argmax(dim=-1)
            all_top1 += int((top1 == tgt).sum().item())
            all_top10 += int((top10 == tgt.unsqueeze(1)).any(dim=1).sum().item())
            if channel == "swap_paired":
                neighbor_tgt = value_token_ids_all[(slots + 1) % n]
                all_paired_flip += int((top1 == neighbor_tgt).sum().item())
                all_paired_count += int(tgt.numel())
        result = {
            "top1": all_top1 / all_count,
            "top10": all_top10 / all_count,
            "nll": all_nll / all_count,
            "n": all_count,
        }
        if channel == "swap_paired" and all_paired_count > 0:
            result["paired_flip_rate"] = all_paired_flip / all_paired_count
        return result

    metrics = {
        "bank_inject_oracle": run("bank_inject_oracle"),
        "bank_inject_retrieved": run("bank_inject_retrieved"),
        "no_memory": run("no_memory"),
        "swap_paired": run("swap_paired"),
        "address_retrieval_recall_at_1": retrieval_recall_at_1,
    }
    return metrics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-facts", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scale on bank injection vector at answer position.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--key-dim", type=int, default=256)
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-hard-negatives", type=int, default=0,
                        help="Per step, also sample K extra global slots as InfoNCE negatives.")
    parser.add_argument("--dataset", default="synthetic_colors",
                        choices=["synthetic_colors", "lama_curated"],
                        help="Source of (address, value) facts.")
    parser.add_argument("--lama-jsonl", default="scripts/data/lama_curated.jsonl",
                        help="Path to LAMA curated JSONL (used when --dataset=lama_curated).")
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage8] loading {args.model} on {args.device} ({args.dtype})", flush=True)
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.eval()
    device = bundle.device
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")

    facts = build_facts(tokenizer, n_facts=args.n_facts, seed=args.seed) if args.dataset == "synthetic_colors" else build_facts_lama(tokenizer, REPO_ROOT / args.lama_jsonl, seed=args.seed, n_facts=args.n_facts)
    print(f"[stage8] dataset={args.dataset} built {len(facts)} facts; hidden={hidden}", flush=True)

    bank = FastWeightBank(n_slots=len(facts), hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)
    writer = Writer(hidden=hidden).to(device=device, dtype=torch.float32)
    key_proj = KeyProjector(hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)

    t0 = time.time()
    history = train(
        model=model,
        tokenizer=tokenizer,
        bank=bank,
        writer=writer,
        key_proj=key_proj,
        facts=facts,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        seed=args.seed,
        retrieval_loss_weight=args.retrieval_loss_weight,
        retrieval_temperature=args.retrieval_temperature,
        retrieval_hard_negatives=args.retrieval_hard_negatives,
    )
    train_secs = time.time() - t0

    metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        bank=bank,
        writer=writer,
        key_proj=key_proj,
        facts=facts,
        device=device,
        alpha=args.alpha,
        eval_batch_size=args.eval_batch_size,
    )

    g1_pass = metrics["bank_inject_retrieved"]["top1"] >= 0.80
    g6_pass = metrics["no_memory"]["top1"] <= 0.05
    g5_swap = metrics["swap_paired"].get("paired_flip_rate", 0.0)
    g5_pass = g5_swap >= 0.80
    g_retr_pass = metrics["address_retrieval_recall_at_1"] >= 0.95

    summary = {
        "stage": "8.1_closed_book_pilot",
        "args": vars(args),
        "n_facts": len(facts),
        "hidden_size": hidden,
        "train_seconds": train_secs,
        "training_history": [s.__dict__ for s in history],
        "metrics": metrics,
        "gates": {
            "G1_closed_book_retrieved_top1_ge_0.80": {"value": metrics["bank_inject_retrieved"]["top1"], "pass": g1_pass},
            "G6_no_memory_top1_le_0.05": {"value": metrics["no_memory"]["top1"], "pass": g6_pass},
            "G5_swap_paired_flip_ge_0.80": {"value": g5_swap, "pass": g5_pass},
            "GR_address_recall_at_1_ge_0.95": {"value": metrics["address_retrieval_recall_at_1"], "pass": g_retr_pass},
        },
        "writer_params": sum(p.numel() for p in writer.parameters() if p.requires_grad),
        "key_proj_params": sum(p.numel() for p in key_proj.parameters() if p.requires_grad),
    }

    summary_path = report_dir / "delta_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stage8] summary -> {summary_path}", flush=True)

    print("\n[stage8] === RESULTS ===")
    print(f"  closed-book bank_inject_retrieved top1 = {metrics['bank_inject_retrieved']['top1']:.3f}  (G1: >=0.80 -> {'PASS' if g1_pass else 'FAIL'})")
    print(f"  closed-book bank_inject_oracle    top1 = {metrics['bank_inject_oracle']['top1']:.3f}  (sanity upper bound)")
    print(f"  address retrieval recall@1            = {metrics['address_retrieval_recall_at_1']:.3f}  (GR: >=0.95 -> {'PASS' if g_retr_pass else 'FAIL'})")
    print(f"  no_memory baseline top1                = {metrics['no_memory']['top1']:.3f}  (G6: <=0.05 -> {'PASS' if g6_pass else 'FAIL'})")
    print(f"  swap_paired paired-flip rate           = {g5_swap:.3f}  (G5: >=0.80 -> {'PASS' if g5_pass else 'FAIL'})")
    print(f"  closed-book NLL (retrieved)            = {metrics['bank_inject_retrieved']['nll']:.3f}")
    print(f"  no_memory NLL                          = {metrics['no_memory']['nll']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
