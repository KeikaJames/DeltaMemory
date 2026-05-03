"""Address-encoder variants for Stage 9.

All encoders consume the address span tokens (and optionally the read
prompt for "prompt-hidden" encoders that derive the query from the
frozen base model's last hidden state). They produce a single dense
vector per fact, which the Stage 8 ``KeyProjector`` then maps into the
trainable retrieval space.

Variants
--------
- ``mean_pool``        — Stage 8 v3 baseline. Mean-pool input embeddings
                         of address tokens. No learnable parameters.
- ``attn_pool``        — Learnable attention pool over address tokens.
- ``multilayer``       — Run the address tokens through the frozen base
                         (no padding / no read prompt) and concat / pool
                         hidden states from multiple layers.
- ``prompt_hidden``    — Run the read prompt through the frozen base and
                         take the last layer hidden state at the address
                         token position. This is the "query-conditioned"
                         variant: the writer/key share Gemma's 26-layer
                         contextual feature.
- ``residual_mlp``     — Wraps mean_pool with a 2-layer residual MLP for
                         a deeper non-linear key projection prior to
                         ``KeyProjector``.

For the encoders that don't have learnable parameters (``mean_pool``,
``multilayer``, ``prompt_hidden``) the tensor returned is the encoded
address. For ``attn_pool`` / ``residual_mlp`` learnable params live
on the encoder module and must be added to the optimiser.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize_addresses(tokenizer, addresses: List[str], device, max_len: int = 32):
    enc = tokenizer(
        addresses, padding=True, truncation=True, max_length=max_len,
        add_special_tokens=False, return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _tokenize_prompts(tokenizer, prompts: List[str], device, max_len: int = 96):
    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=max_len,
        add_special_tokens=True, return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class AddressEncoder(nn.Module):
    """Common interface."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def output_dim(self) -> int:
        return self.hidden

    def encode(
        self,
        model,
        tokenizer,
        addresses: List[str],
        prompts: List[str] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Mean-pool baseline
# ---------------------------------------------------------------------------

class MeanPoolEncoder(AddressEncoder):
    """Stage 8 v3 baseline. No learnable params."""

    def encode(self, model, tokenizer, addresses, prompts=None):
        device = next(model.parameters()).device
        ids, mask = _tokenize_addresses(tokenizer, addresses, device)
        embed = model.get_input_embeddings()
        e = embed(ids)
        m = mask.unsqueeze(-1).float()
        pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return pooled.float()


# ---------------------------------------------------------------------------
# 2. Attention pool
# ---------------------------------------------------------------------------

class AttnPoolEncoder(AddressEncoder):
    """Learnable single-query attention over address-token embeddings."""

    def __init__(self, hidden: int):
        super().__init__(hidden)
        self.query = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.proj_k = nn.Linear(hidden, hidden, bias=False)
        self.proj_v = nn.Linear(hidden, hidden, bias=False)

    def encode(self, model, tokenizer, addresses, prompts=None):
        device = next(model.parameters()).device
        ids, mask = _tokenize_addresses(tokenizer, addresses, device)
        embed = model.get_input_embeddings()
        e = embed(ids).float()
        k = self.proj_k(e)
        v = self.proj_v(e)
        q = self.query.expand(e.size(0), -1).unsqueeze(1)
        scores = (q @ k.transpose(1, 2)).squeeze(1) / (self.hidden ** 0.5)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * v).sum(dim=1)


# ---------------------------------------------------------------------------
# 3. Multi-layer hidden state of address tokens (no read prompt)
# ---------------------------------------------------------------------------

class MultiLayerEncoder(AddressEncoder):
    """Forward address tokens through the frozen base, mean-pool the last
    hidden state. (Multi-layer concat is overkill at this stage; using
    just the last layer already gives Gemma's full contextual features.)
    """

    def encode(self, model, tokenizer, addresses, prompts=None):
        device = next(model.parameters()).device
        ids, mask = _tokenize_addresses(tokenizer, addresses, device)
        with torch.no_grad():
            out = model.model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=False,
                use_cache=False,
            )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return pooled.float()


# ---------------------------------------------------------------------------
# 4. Prompt-hidden (query-conditioned)
# ---------------------------------------------------------------------------

class PromptHiddenEncoder(AddressEncoder):
    """Run the read prompt through the frozen base and take the final
    real-token hidden state as the address representation.

    This is the cleanest "query-conditioned" variant: the writer/key
    share Gemma's full 26-layer contextualised feature for the *exact*
    point where the model will try to recall the value. It works
    uniformly for synthetic prompts ("Atlas slot S-00042-...\\n...the
    value is") and natural ones ("The capital of France is"), without
    needing to locate an address span.
    """

    def encode(self, model, tokenizer, addresses, prompts=None):
        if prompts is None:
            raise ValueError("PromptHiddenEncoder requires read prompts")
        device = next(model.parameters()).device
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=96,
                        add_special_tokens=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        last = mask.sum(dim=1) - 1
        with torch.no_grad():
            out = model.model(
                input_ids=ids, attention_mask=mask,
                output_hidden_states=False, use_cache=False,
            )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        B, L, H = h.shape
        idx = last.view(B, 1, 1).expand(B, 1, H)
        return h.gather(1, idx).squeeze(1).float()


# ---------------------------------------------------------------------------
# 5. Residual MLP wrapper around mean-pool
# ---------------------------------------------------------------------------

class ResidualMLPEncoder(AddressEncoder):
    """Mean-pool + 2-layer residual MLP. Cheap learnable upgrade that
    keeps the same input but adds non-linear capacity before the
    KeyProjector."""

    def __init__(self, hidden: int, mult: int = 2):
        super().__init__(hidden)
        self.proj1 = nn.Linear(hidden, hidden * mult)
        self.proj2 = nn.Linear(hidden * mult, hidden)
        self.norm = nn.LayerNorm(hidden)

    def encode(self, model, tokenizer, addresses, prompts=None):
        device = next(model.parameters()).device
        ids, mask = _tokenize_addresses(tokenizer, addresses, device)
        embed = model.get_input_embeddings()
        e = embed(ids).float()
        m = mask.unsqueeze(-1).float()
        pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        h = self.norm(pooled)
        h = h + self.proj2(F.gelu(self.proj1(h)))
        return h


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "mean_pool": MeanPoolEncoder,
    "attn_pool": AttnPoolEncoder,
    "multilayer": MultiLayerEncoder,
    "prompt_hidden": PromptHiddenEncoder,
    "residual_mlp": ResidualMLPEncoder,
}


def build_encoder(name: str, hidden: int) -> AddressEncoder:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown encoder {name!r}; valid: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    return cls(hidden)
