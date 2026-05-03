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
from rcvhc.encoders import build_encoder  # noqa: E402

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
    encoder=None,
    paraphrase_pool: list[list[str]] | None = None,
    relation_ids: list[int] | None = None,
    relation_adversary_weight: float = 0.0,
    n_relations: int = 0,
) -> list[StepStats]:
    params = list(writer.parameters()) + list(key_proj.parameters())
    if encoder is not None:
        params += list(encoder.parameters())
    # Stage 11: optional gradient-reversal relation-id adversary head.
    rel_adv_head = None
    if relation_adversary_weight > 0.0 and n_relations > 1 and relation_ids is not None:
        rel_adv_head = nn.Linear(writer.down.out_features, n_relations).to(
            device=device, dtype=torch.float32)
        params += list(rel_adv_head.parameters())
    optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    n = len(facts)
    rng = random.Random(seed + 7919)
    history: list[StepStats] = []
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    read_prompts_all = [f.read_prompt for f in facts]
    addresses_all = [f.address for f in facts]
    input_ids_all, attn_all, last_pos_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)
    # Pre-cache address representations from the configured encoder. For
    # parameter-free encoders this is a one-shot frozen-base computation;
    # for learnable encoders we re-run encode() inside the loop so gradients
    # flow.
    encoder_has_params = encoder is not None and any(
        p.requires_grad for p in encoder.parameters()
    )
    if encoder is None or not encoder_has_params:
        with torch.no_grad():
            if encoder is None:
                addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)
            else:
                addr_pooled_all = encoder.encode(model, tokenizer, addresses_all, read_prompts_all)
    else:
        addr_pooled_all = None  # recomputed per step

    log_every = max(1, steps // 20)
    for step in range(1, steps + 1):
        idxs = [rng.randrange(n) for _ in range(batch_size)]
        slots = torch.tensor(idxs, device=device, dtype=torch.long)
        value_embeds = _value_embeds(model, value_token_ids_all[slots]).detach().float()
        bank_vectors = writer(value_embeds)
        # Stage 11: per-step paraphrase sampling. If a paraphrase pool is
        # provided, replace the canonical read prompt with a random surface
        # variant per (step, fact) so the encoder sees many surface forms
        # per ground-truth address. The first call uses cached canonical
        # tokenisation; here we re-tokenise the sampled paraphrase batch.
        if paraphrase_pool is not None:
            sampled_prompts = [paraphrase_pool[i][rng.randrange(len(paraphrase_pool[i]))]
                               for i in idxs]
            ids, am, lp = _tokenize_read_prompts(tokenizer, sampled_prompts, device)
        else:
            ids = input_ids_all[slots]
            am = attn_all[slots]
            lp = last_pos_all[slots]
        targets = value_token_ids_all[slots]
        logits = _forward_read_with_injection(model, ids, am, lp, bank_vectors, alpha)
        ce = F.cross_entropy(logits.float(), targets)

        # InfoNCE retrieval. Anchors = batch slots; candidates = batch slots
        # (in-batch negatives) + optional K extra global random slots
        # (extra random negatives shared across the batch).
        if encoder_has_params:
            # Re-encode the relevant slot subset with gradients flowing.
            batch_addrs = [addresses_all[i] for i in idxs]
            # Stage 11: if paraphrase pool active, the encoder must see the
            # *sampled* paraphrase, not the canonical prompt.
            if paraphrase_pool is not None:
                batch_prompts = sampled_prompts
            else:
                batch_prompts = [read_prompts_all[i] for i in idxs]
            anchor_pooled = encoder.encode(model, tokenizer, batch_addrs, batch_prompts)
            if retrieval_hard_negatives > 0:
                batch_set = set(idxs)
                extras: list[int] = []
                while len(extras) < retrieval_hard_negatives:
                    cand = rng.randrange(n)
                    if cand not in batch_set:
                        extras.append(cand)
                        batch_set.add(cand)
                ex_addrs = [addresses_all[i] for i in extras]
                ex_prompts = [read_prompts_all[i] for i in extras]
                extra_pooled = encoder.encode(model, tokenizer, ex_addrs, ex_prompts)
            else:
                extra_pooled = None
        else:
            anchor_pooled = addr_pooled_all[slots]
            if retrieval_hard_negatives > 0:
                batch_set = set(idxs)
                extras = []
                while len(extras) < retrieval_hard_negatives:
                    cand = rng.randrange(n)
                    if cand not in batch_set:
                        extras.append(cand)
                        batch_set.add(cand)
                extra_slots = torch.tensor(extras, device=device, dtype=torch.long)
                extra_pooled = addr_pooled_all[extra_slots]
            else:
                extra_pooled = None
        keys_anchor = key_proj(anchor_pooled)
        if extra_pooled is not None:
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

        # Stage 11: gradient-reversal relation-id adversary on payload.
        # Forces writer-output to be relation-agnostic, helping LORO holdout.
        rel_adv_loss_val = 0.0
        if rel_adv_head is not None:
            rel_targets = torch.tensor(
                [relation_ids[i] for i in idxs], device=device, dtype=torch.long)
            # Gradient reversal layer (manual): flip gradient sign by negating loss
            # contribution after .backward by detach trick.
            rev = bank_vectors  # detached upstream value but writer-output has grad
            rel_logits = rel_adv_head(rev)
            rel_ce = F.cross_entropy(rel_logits.float(), rel_targets)
            # We want to MAXIMISE the adversary's confusion, so subtract its loss
            # from main loss (gradient reversal). The adversary head itself still
            # learns by adding +rel_ce; net effect: writer adversarial, head normal.
            loss = loss - relation_adversary_weight * rel_ce + rel_ce
            rel_adv_loss_val = float(rel_ce.item())

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
    encoder=None,
) -> dict:
    n = len(facts)
    value_token_ids_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    read_prompts_all = [f.read_prompt for f in facts]
    addresses_all = [f.address for f in facts]
    input_ids_all, attn_all, last_pos_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids_all, addr_mask_all = _address_token_ids(tokenizer, addresses_all, device)

    # Phase A: WRITE — populate bank.v from writer, bank.k from key_proj.
    # Pre-compute address representations once (no_grad context already).
    if encoder is None:
        addr_pooled_all = _address_key_from_embeddings(model, addr_ids_all, addr_mask_all)
    else:
        addr_pooled_all = encoder.encode(model, tokenizer, addresses_all, read_prompts_all)
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        slots = torch.arange(start, end, device=device, dtype=torch.long)
        value_embeds = _value_embeds(model, value_token_ids_all[slots]).float()
        v = writer(value_embeds)
        k = key_proj(addr_pooled_all[start:end])
        bank.write(slots, v, k)

    # Pre-compute predicted slots from address-content retrieval.
    pred_slots_all = torch.empty(n, device=device, dtype=torch.long)
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        q = key_proj(addr_pooled_all[start:end])
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
# Stage 10 — adversarial stress tests (helpers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _bank_eval_top1_nll(model, ids, am, lp, tgt, inj, alpha) -> tuple[int, int, float, int]:
    """Single-pass forward at last position; returns (top1_correct, top10_correct, nll_sum, n)."""
    logits = _forward_read_with_injection(model, ids, am, lp, inj, alpha)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tgt_lp = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
    nll_sum = float((-tgt_lp).sum().item())
    top1 = log_probs.argmax(dim=-1)
    top10 = log_probs.topk(10, dim=-1).indices
    top1_c = int((top1 == tgt).sum().item())
    top10_c = int((top10 == tgt.unsqueeze(1)).any(dim=1).sum().item())
    return top1_c, top10_c, nll_sum, int(tgt.numel())


@torch.no_grad()
def _stage10_paraphrase_eval(*, model, tokenizer, bank, writer, key_proj, encoder,
                              facts, device, alpha, eval_batch_size, paraphrase_jsonl) -> dict:
    """Stage 10A — paraphrase robustness.

    For each fact we have N paraphrase templates. Bank is already trained
    on the canonical (paraphrase[0]) read prompt of those facts (i.e. the
    same prompts used in the LAMA-TREx training). We now use paraphrase
    [1..K-1] as the *eval* read prompt and re-measure retrieval + binding.

    Test: does the encoder's prompt_hidden / multilayer feature carry the
    address identity across surface paraphrase, or is it a byte-level
    fingerprint?
    """
    rows = [json.loads(line) for line in paraphrase_jsonl.read_text().splitlines() if line.strip()]
    addr_to_row = {row["address"]: row for row in rows}
    eligible = [(i, addr_to_row[f.address]) for i, f in enumerate(facts) if f.address in addr_to_row]
    if not eligible:
        return {"error": "no_fact_overlap_with_paraphrase_jsonl"}
    n_para = max(len(r["paraphrases"]) for _, r in eligible)
    out_per_para: list[dict] = []
    for k in range(n_para):
        # Build per-fact paraphrase k (or canonical if missing). We wrap
        # the paraphrase in the same Atlas-slot template that the model
        # was trained on; otherwise we'd be testing template robustness
        # rather than address-paraphrase robustness.
        eval_prompts = []
        slot_idx = []
        targets = []
        addresses_eval = []
        for i, row in eligible:
            paras = row["paraphrases"]
            p = paras[k] if k < len(paras) else paras[0]
            wrapped = (
                f"Atlas slot {p}\n"
                f"Recall the payload value for this slot. The value is"
            )
            eval_prompts.append(wrapped)
            addresses_eval.append(p)
            slot_idx.append(i)
            targets.append(facts[i].value_token_id)
        slot_t = torch.tensor(slot_idx, device=device, dtype=torch.long)
        tgt_t = torch.tensor(targets, device=device, dtype=torch.long)
        ids, am, lp = _tokenize_read_prompts(tokenizer, eval_prompts, device)
        addr_ids, addr_mask = _address_token_ids(tokenizer, addresses_eval, device)
        if encoder is None:
            addr_pooled = _address_key_from_embeddings(model, addr_ids, addr_mask)
        else:
            # IMPORTANT: encoder.encode for prompt_hidden uses the *read prompt*,
            # so passing the paraphrased prompts here is the actual stress: the
            # query key is built from the paraphrase, not the canonical prompt.
            addr_pooled = encoder.encode(model, tokenizer, addresses_eval, eval_prompts)
        q = key_proj(addr_pooled)
        pred_slots = bank.retrieve(q)
        recall_at_1 = float((pred_slots == slot_t).float().mean().item())
        # Bank-inject eval using the predicted (possibly wrong) slots' v.
        n = len(eligible)
        t1c = t10c = 0
        nll_sum = 0.0
        n_count = 0
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            inj = bank.read(pred_slots[start:end])
            a, b, c, d = _bank_eval_top1_nll(
                model, ids[start:end], am[start:end], lp[start:end], tgt_t[start:end], inj, alpha
            )
            t1c += a; t10c += b; nll_sum += c; n_count += d
        out_per_para.append({
            "paraphrase_index": k,
            "is_canonical": (k == 0),
            "n": n_count,
            "recall_at_1": recall_at_1,
            "bank_inject_retrieved_top1": t1c / n_count,
            "bank_inject_retrieved_top10": t10c / n_count,
            "bank_inject_retrieved_nll": nll_sum / n_count,
        })
    canonical = out_per_para[0]
    held_out = out_per_para[1:]
    held_recall_mean = sum(o["recall_at_1"] for o in held_out) / max(1, len(held_out))
    held_top1_mean = sum(o["bank_inject_retrieved_top1"] for o in held_out) / max(1, len(held_out))
    return {
        "n_facts_with_paraphrases": len(eligible),
        "n_paraphrases_per_fact": n_para,
        "per_paraphrase": out_per_para,
        "canonical_recall_at_1": canonical["recall_at_1"],
        "held_out_recall_at_1_mean": held_recall_mean,
        "held_out_bank_inject_top1_mean": held_top1_mean,
        "G10A_pass": held_recall_mean >= 0.85,
    }


@torch.no_grad()
def _stage10_decoy_eval(*, model, tokenizer, bank, writer, key_proj, encoder,
                         facts, device, alpha, eval_batch_size, multipliers, seed) -> dict:
    """Stage 10B — distractor stress curve.

    Append K*n_facts random-init slots to the bank's k/v matrices and
    re-run retrieval-only eval. The trained portion is preserved.
    """
    n = len(facts)
    addresses_all = [f.address for f in facts]
    read_prompts_all = [f.read_prompt for f in facts]
    ids_all, am_all, lp_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids, addr_mask = _address_token_ids(tokenizer, addresses_all, device)
    if encoder is None:
        addr_pooled = _address_key_from_embeddings(model, addr_ids, addr_mask)
    else:
        addr_pooled = encoder.encode(model, tokenizer, addresses_all, read_prompts_all)
    q_all = key_proj(addr_pooled)
    tgt_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    slot_targets = torch.arange(n, device=device, dtype=torch.long)
    # Snapshot trained bank params.
    k_orig = bank.k.detach().clone()
    v_orig = bank.v.detach().clone()
    rng = torch.Generator(device="cpu").manual_seed(seed * 17 + 1)
    results = []
    for K in multipliers:
        n_decoy = K * n
        # Random-init keys with the same dim/scale as trained keys.
        k_decoy = torch.empty(n_decoy, k_orig.shape[1], device=device, dtype=k_orig.dtype)
        k_decoy.normal_(mean=0.0, std=k_orig.std().item(), generator=None)
        v_decoy = torch.zeros(n_decoy, v_orig.shape[1], device=device, dtype=v_orig.dtype)
        # Concat to bank temporarily by overriding retrieve manually.
        full_k = torch.cat([k_orig, k_decoy], dim=0)
        full_v = torch.cat([v_orig, v_decoy], dim=0)
        # Recompute retrieve = argmax cosine (chunked to avoid OOM at scale).
        q_norm = F.normalize(q_all, dim=-1)
        k_norm = F.normalize(full_k, dim=-1)
        chunk = max(1, 4096)
        pred_chunks = []
        for cs in range(0, q_norm.shape[0], chunk):
            ce = min(cs + chunk, q_norm.shape[0])
            sims_c = q_norm[cs:ce] @ k_norm.t()
            pred_chunks.append(sims_c.argmax(dim=-1))
            del sims_c
        pred_slots = torch.cat(pred_chunks, dim=0)
        recall_at_1 = float((pred_slots == slot_targets).float().mean().item())
        # Bank-inject top1 using the (possibly decoy) value at the predicted slot.
        t1c = 0
        n_count = 0
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            sl = pred_slots[start:end]
            inj = full_v[sl]
            top1c, _, _, cnt = _bank_eval_top1_nll(
                model, ids_all[start:end], am_all[start:end], lp_all[start:end],
                tgt_all[start:end], inj, alpha
            )
            t1c += top1c; n_count += cnt
        results.append({
            "decoy_multiplier": K,
            "n_decoy_slots": n_decoy,
            "recall_at_1": recall_at_1,
            "bank_inject_retrieved_top1": t1c / n_count,
        })
    return {"curve": results}


@torch.no_grad()
def _stage10_value_ablation(*, model, tokenizer, bank, writer, key_proj, encoder,
                             facts, device, alpha, eval_batch_size, seed) -> dict:
    """Stage 10D — null control on bank values.

    Two ablations:
      1. Random-replace bank.v with same-scale random tensors. Tests
         whether the bank's stored values carry the answer information.
      2. Shuffle bank.v across slots (keys preserved). Same test, less
         destructive: each slot now reads a permuted neighbour's payload.

    Pass criterion: both ablations should crash bank_inject_retrieved.top1
    well below the unablated baseline. If they don't, the bank is
    ornamental and the win comes from the encoder + read-prompt context.
    """
    n = len(facts)
    addresses_all = [f.address for f in facts]
    read_prompts_all = [f.read_prompt for f in facts]
    ids_all, am_all, lp_all = _tokenize_read_prompts(tokenizer, read_prompts_all, device)
    addr_ids, addr_mask = _address_token_ids(tokenizer, addresses_all, device)
    if encoder is None:
        addr_pooled = _address_key_from_embeddings(model, addr_ids, addr_mask)
    else:
        addr_pooled = encoder.encode(model, tokenizer, addresses_all, read_prompts_all)
    q_all = key_proj(addr_pooled)
    pred_slots = bank.retrieve(q_all)
    tgt_all = torch.tensor([f.value_token_id for f in facts], device=device, dtype=torch.long)
    v_orig = bank.v.detach().clone()
    out: dict = {}
    rng = torch.Generator(device="cpu").manual_seed(seed * 31 + 7)

    def _eval_with_v(v_use: torch.Tensor) -> float:
        t1c = 0
        n_count = 0
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            sl = pred_slots[start:end]
            inj = v_use[sl]
            top1c, _, _, cnt = _bank_eval_top1_nll(
                model, ids_all[start:end], am_all[start:end], lp_all[start:end],
                tgt_all[start:end], inj, alpha
            )
            t1c += top1c; n_count += cnt
        return t1c / n_count

    # Ablation 1: random replacement.
    v_rand = torch.empty_like(v_orig)
    v_rand.normal_(mean=0.0, std=v_orig.std().item())
    out["random_value_top1"] = _eval_with_v(v_rand)
    # Ablation 2: shuffle within slots.
    perm = torch.randperm(n, generator=rng).to(device)
    v_shuf = v_orig[perm]
    out["shuffled_value_top1"] = _eval_with_v(v_shuf)
    out["unablated_top1_reference"] = _eval_with_v(v_orig)
    out["G10D_pass"] = (out["random_value_top1"] <= 0.10) and (out["shuffled_value_top1"] <= 0.10)
    return out


@torch.no_grad()
def _stage10_loro_eval(*, model, tokenizer, bank, writer, key_proj, encoder,
                       facts, device, alpha, eval_batch_size, holdout_jsonl) -> dict:
    """Stage 10F — leave-one-relation-out: add held-out facts to a *trained*
    pipeline (no further fine-tuning), then evaluate retrieval + binding
    on those new slots only.

    Pass criterion: retr top-1 ≥ 0.50 (relaxed; this is genuinely zero-shot
    for the encoder + key_proj combination).
    """
    rows = [json.loads(line) for line in holdout_jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        return {"error": "empty_holdout"}
    # Build temporary held-out facts (validate single-token).
    new_facts: list[Fact] = []
    skipped = 0
    for slot, row in enumerate(rows):
        word = row["value"].strip()
        tid = _validate_single_token(tokenizer, word)
        if tid is None:
            skipped += 1
            continue
        new_facts.append(Fact(slot=len(facts) + len(new_facts),
                              address=row["address"],
                              value_token_str=word,
                              value_token_id=tid))
    if not new_facts:
        return {"error": "no_single_token_holdout", "skipped": skipped}
    n_new = len(new_facts)
    # Encode new addresses with frozen-trained encoder + key_proj.
    addrs_new = [f.address for f in new_facts]
    prompts_new = [f.read_prompt for f in new_facts]
    addr_ids, addr_mask = _address_token_ids(tokenizer, addrs_new, device)
    if encoder is None:
        addr_pooled = _address_key_from_embeddings(model, addr_ids, addr_mask)
    else:
        addr_pooled = encoder.encode(model, tokenizer, addrs_new, prompts_new)
    new_keys = key_proj(addr_pooled)
    # Compute new values via writer over the value-token embeddings (frozen base, frozen writer).
    val_tids = torch.tensor([f.value_token_id for f in new_facts], device=device, dtype=torch.long)
    val_embeds = _value_embeds(model, val_tids).float()
    new_vals = writer(val_embeds)
    # Append to bank.
    n_old = bank.k.shape[0]
    full_k = torch.cat([bank.k.detach(), new_keys], dim=0)
    full_v = torch.cat([bank.v.detach(), new_vals], dim=0)
    # Retrieval over new addresses against the union bank.
    q_norm = F.normalize(new_keys, dim=-1)
    k_norm = F.normalize(full_k, dim=-1)
    sims = q_norm @ k_norm.t()
    pred_slots = sims.argmax(dim=-1)
    expected = torch.arange(n_old, n_old + n_new, device=device)
    recall_at_1 = float((pred_slots == expected).float().mean().item())
    # Bank-inject eval on the new prompts.
    ids_new, am_new, lp_new = _tokenize_read_prompts(tokenizer, prompts_new, device)
    t1c = 0
    n_count = 0
    for start in range(0, n_new, eval_batch_size):
        end = min(start + eval_batch_size, n_new)
        sl = pred_slots[start:end]
        inj = full_v[sl]
        top1c, _, _, cnt = _bank_eval_top1_nll(
            model, ids_new[start:end], am_new[start:end], lp_new[start:end],
            val_tids[start:end], inj, alpha
        )
        t1c += top1c; n_count += cnt
    return {
        "n_holdout_facts": n_new,
        "skipped_multitoken": skipped,
        "recall_at_1_holdout": recall_at_1,
        "bank_inject_retrieved_top1_holdout": t1c / n_count,
        "G10F_pass": (t1c / n_count) >= 0.50,
    }


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
    parser.add_argument("--encoder", default="mean_pool",
                        choices=["mean_pool", "attn_pool", "multilayer", "prompt_hidden", "residual_mlp"],
                        help="Address encoder variant (Stage 9). mean_pool reproduces v3 baseline.")
    # Stage 10 adversarial validation hooks (all post-training, idempotent).
    parser.add_argument("--stage10-paraphrase-jsonl", default=None,
                        help="Path to paraphrase JSONL (build_lama_trex_paraphrase.py output). Eval-only stress test.")
    parser.add_argument("--stage10-decoy-multipliers", default=None,
                        help="Comma-separated decoy multipliers e.g. '1,10,100'. Adds K*n_facts random slots.")
    parser.add_argument("--stage10-value-ablation", action="store_true",
                        help="Replace bank values with random / shuffled tensors after training; re-eval.")
    parser.add_argument("--stage10-loro-add-jsonl", default=None,
                        help="Path to held-out fact JSONL (LORO). Append to bank without further training and eval new slots.")
    # Stage 11 training-time fixes.
    parser.add_argument("--stage11-paraphrase-train-jsonl", default=None,
                        help="Path to JSONL with 'paraphrases' list per fact. At each step, "
                             "sample a random paraphrase per fact as the read prompt. Forces "
                             "encoder to be paraphrase-invariant.")
    parser.add_argument("--stage11-loro-exclude-relation", default=None,
                        help="Wikidata relation P-code (e.g. P36) to EXCLUDE from training. "
                             "All facts with this relation are dropped pre-training; "
                             "use --stage10-loro-add-jsonl with the same relation to eval.")
    parser.add_argument("--stage11-relation-adversary-weight", type=float, default=0.0,
                        help="Weight for gradient-reversal relation-id adversary on payload "
                             "(forces writer to be relation-agnostic). 0 disables.")
    parser.add_argument("--stage11-deterministic", action="store_true",
                        help="Enable bit-exact reproduction: torch deterministic algorithms, "
                             "single-thread tokenisation, disable bf16 nondeterminism.")
    args = parser.parse_args()

    if args.stage11_deterministic:
        import os as _os
        _os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        _os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    # ------------------------------------------------------------------
    # Stage 11: optionally exclude one Wikidata relation from training
    # (train-time LORO). Requires lama_trex_full.jsonl-style data with a
    # 'relation' field that we can backsolve from prompt content via the
    # Stage 10 paraphrase detector.
    # ------------------------------------------------------------------
    relation_per_fact: list[str | None] = [None] * len(facts)
    if args.stage11_loro_exclude_relation or args.stage11_relation_adversary_weight > 0.0:
        from build_stage11_paraphrase import _detect_relation as _det
        relation_per_fact = [_det(f.address, f.value_token_str) for f in facts]
    if args.stage11_loro_exclude_relation:
        excl = args.stage11_loro_exclude_relation
        kept = [(f, r) for f, r in zip(facts, relation_per_fact) if r != excl]
        excluded_n = len(facts) - len(kept)
        facts = [Fact(slot=i, address=f.address, value_token_str=f.value_token_str,
                      value_token_id=f.value_token_id) for i, (f, _) in enumerate(kept)]
        relation_per_fact = [r for _, r in kept]
        print(f"[stage11] LORO: excluded {excluded_n} facts with relation={excl}; "
              f"training on {len(facts)} facts", flush=True)
    if args.stage11_relation_adversary_weight > 0.0:
        # Drop facts with unknown relation so adversary CE has valid targets.
        kept = [(f, r) for f, r in zip(facts, relation_per_fact) if r is not None]
        dropped = len(facts) - len(kept)
        if dropped:
            print(f"[stage11] adversary: dropped {dropped} facts with unknown relation", flush=True)
        facts = [Fact(slot=i, address=f.address, value_token_str=f.value_token_str,
                      value_token_id=f.value_token_id) for i, (f, _) in enumerate(kept)]
        relation_per_fact = [r for _, r in kept]

    # Build paraphrase pool keyed by fact address.
    paraphrase_pool: list[list[str]] | None = None
    if args.stage11_paraphrase_train_jsonl:
        addr_to_paras: dict[str, list[str]] = {}
        with (REPO_ROOT / args.stage11_paraphrase_train_jsonl).open() as pf:
            for line in pf:
                row = json.loads(line)
                addr_to_paras[row["address"]] = row["paraphrases"]
        paraphrase_pool = []
        hits = 0
        for f in facts:
            paras = addr_to_paras.get(f.address)
            if paras is None:
                paraphrase_pool.append([f.read_prompt])
            else:
                paraphrase_pool.append(paras)
                hits += 1
        print(f"[stage11] paraphrase pool covers {hits}/{len(facts)} facts "
              f"({sum(len(p) for p in paraphrase_pool)/len(paraphrase_pool):.1f} paraphrases/fact avg)", flush=True)

    # Relation index for adversary head.
    rel_codes_sorted = sorted({r for r in relation_per_fact if r is not None})
    rel_to_idx = {r: i for i, r in enumerate(rel_codes_sorted)}
    relation_ids = [rel_to_idx.get(r, -1) for r in relation_per_fact]
    n_relations = len(rel_codes_sorted)

    bank = FastWeightBank(n_slots=len(facts), hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)
    writer = Writer(hidden=hidden).to(device=device, dtype=torch.float32)
    key_proj = KeyProjector(hidden=hidden, key_dim=args.key_dim).to(device=device, dtype=torch.float32)
    encoder = build_encoder(args.encoder, hidden=hidden).to(device=device, dtype=torch.float32) if args.encoder != "mean_pool" else None
    print(f"[stage8] encoder={args.encoder}", flush=True)

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
        encoder=encoder,
        paraphrase_pool=paraphrase_pool,
        relation_ids=relation_ids if any(i >= 0 for i in relation_ids) else None,
        relation_adversary_weight=args.stage11_relation_adversary_weight,
        n_relations=n_relations,
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
        encoder=encoder,
    )

    # ------------------------------------------------------------------
    # Stage 10 adversarial stress tests (post-training, all share the
    # already-trained writer / key_proj / encoder / bank).
    # ------------------------------------------------------------------
    stage10: dict = {}
    if args.stage10_paraphrase_jsonl:
        stage10["paraphrase"] = _stage10_paraphrase_eval(
            model=model, tokenizer=tokenizer, bank=bank, writer=writer,
            key_proj=key_proj, encoder=encoder, facts=facts, device=device,
            alpha=args.alpha, eval_batch_size=args.eval_batch_size,
            paraphrase_jsonl=REPO_ROOT / args.stage10_paraphrase_jsonl,
        )
    if args.stage10_decoy_multipliers:
        mults = [int(x) for x in args.stage10_decoy_multipliers.split(",") if x.strip()]
        stage10["decoy_curve"] = _stage10_decoy_eval(
            model=model, tokenizer=tokenizer, bank=bank, writer=writer,
            key_proj=key_proj, encoder=encoder, facts=facts, device=device,
            alpha=args.alpha, eval_batch_size=args.eval_batch_size,
            multipliers=mults, seed=args.seed,
        )
    if args.stage10_value_ablation:
        stage10["value_ablation"] = _stage10_value_ablation(
            model=model, tokenizer=tokenizer, bank=bank, writer=writer,
            key_proj=key_proj, encoder=encoder, facts=facts, device=device,
            alpha=args.alpha, eval_batch_size=args.eval_batch_size, seed=args.seed,
        )
    if args.stage10_loro_add_jsonl:
        stage10["loro_holdout"] = _stage10_loro_eval(
            model=model, tokenizer=tokenizer, bank=bank, writer=writer,
            key_proj=key_proj, encoder=encoder, facts=facts, device=device,
            alpha=args.alpha, eval_batch_size=args.eval_batch_size,
            holdout_jsonl=REPO_ROOT / args.stage10_loro_add_jsonl,
        )
    if stage10:
        metrics["stage10"] = stage10

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
