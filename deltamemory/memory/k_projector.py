"""Stage 14A — InfoNCE K-projector.

A per-layer ``nn.Linear(d_h, d_h)`` applied **only to bank keys** at read
time. Trained by InfoNCE on (canonical write prompt, paraphrase query)
positive pairs so that paraphrase Q's land in the same K-space neighborhood
as the canonical write K's.

Invariants
----------
* **Identity init**: untrained projector is exactly ``W = I, b = 0``, which
  means an attached-but-untrained projector is a no-op. All existing
  bit-equality gates therefore continue to hold even after attaching.
* **alpha = 0 invariance**: the patched attention skips the bank branch
  entirely when alpha = 0 or bank is empty, so the projector never runs in
  that case.
* **Bank-only**: the projector is applied to ``mk`` inside the bank scoring
  branch only. Sequence Q and sequence K are never projected.

Usage
-----
    bank = AttnNativeBank(...)
    proj = KProjectorBank.identity_for(model)   # one Linear per attn layer
    bank.k_projector = proj                      # attach
    # train with deltamemory.memory.k_projector.train_infonce(...)

The ``attn_native_bank`` forward hook reads ``bank.k_projector`` and, if
present, calls ``proj(layer_idx, mk)`` before computing scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class KProjectorBank(nn.Module):
    """One ``nn.Linear(d, d)`` per attention layer, identity-initialized."""

    def __init__(self, head_dims: list[int]):
        super().__init__()
        self.head_dims = list(head_dims)
        self.layers = nn.ModuleList()
        for d in self.head_dims:
            lin = nn.Linear(int(d), int(d), bias=True)
            with torch.no_grad():
                lin.weight.copy_(torch.eye(int(d)))
                lin.bias.zero_()
            self.layers.append(lin)

    @classmethod
    def identity_for(cls, model: Any) -> "KProjectorBank":
        """Build an identity projector matching a model's per-layer head_dim."""
        dims: list[int] = []
        for layer in model.model.layers:
            attn = layer.self_attn
            d = getattr(attn, "head_dim", None)
            if d is None:
                d = attn.q_proj.out_features // attn.num_heads
            dims.append(int(d))
        return cls(dims)

    def forward(self, layer_idx: int, mk: torch.Tensor) -> torch.Tensor:
        """Project bank keys for one layer.

        Args:
            layer_idx: attention layer index.
            mk: bank K tensor of shape ``[N, num_kv_heads, head_dim]`` or any
                shape whose last dim equals ``head_dims[layer_idx]``.

        Returns:
            Projected tensor with the same shape and dtype as ``mk``.
        """
        lin = self.layers[layer_idx]
        target_dtype = mk.dtype
        target_device = mk.device
        if lin.weight.dtype != target_dtype or lin.weight.device != target_device:
            lin = lin.to(device=target_device, dtype=target_dtype)
            self.layers[layer_idx] = lin
        return lin(mk)

    def is_identity(self, atol: float = 0.0) -> bool:
        """True iff every layer is exactly the identity (within ``atol``)."""
        for lin, d in zip(self.layers, self.head_dims):
            eye = torch.eye(int(d), dtype=lin.weight.dtype, device=lin.weight.device)
            if (lin.weight - eye).abs().max().item() > atol:
                return False
            if lin.bias.abs().max().item() > atol:
                return False
        return True

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head_dims": self.head_dims, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "KProjectorBank":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        bank = cls(ckpt["head_dims"])
        bank.load_state_dict(ckpt["state_dict"])
        return bank


# ---------------------------------------------------------------------------
# InfoNCE training (lightweight; the Mac MPS path runs in <5 minutes for
# the Gemma-4-E2B 35-layer projector on the train split).
# ---------------------------------------------------------------------------

@dataclass
class InfoNCEBatch:
    """One batch of (write-K, query-Q) pairs per layer.

    All tensors live on the same device. Shapes:
        write_k: [B, num_kv_heads, head_dim]
        query_q: [B, num_q_heads, head_dim]
    The InfoNCE loss treats the diagonal of the (Q, projected K) similarity
    matrix as positive pairs and the off-diagonal as in-batch negatives.
    """

    layer_idx: int
    write_k: torch.Tensor
    query_q: torch.Tensor


def infonce_loss(
    proj: KProjectorBank,
    batch: InfoNCEBatch,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE loss for one layer."""
    pk = proj(batch.layer_idx, batch.write_k)            # [B, Hkv, d]
    pk_pooled = pk.mean(dim=1)                            # [B, d]
    q_pooled = batch.query_q.mean(dim=1)                  # [B, d]
    pk_pooled = nn.functional.normalize(pk_pooled, dim=-1)
    q_pooled = nn.functional.normalize(q_pooled, dim=-1)
    logits = (q_pooled @ pk_pooled.t()) / max(temperature, 1e-6)  # [B, B]
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, targets)
        + nn.functional.cross_entropy(logits.t(), targets)
    )
