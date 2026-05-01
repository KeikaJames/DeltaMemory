"""Raw/Delta Memory writer."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from rcvhc.core.types import AttentionMemoryItem


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RCVHCWriter(nn.Module):
    """Write per-block RawMemory and Delta Q/K/V memory.

    The writer consumes frozen model outputs but its projections remain
    trainable. It never stores full hidden histories or KV cache.
    """

    def __init__(self, hidden_size: int, memory_dim: int, block_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.memory_dim = int(memory_dim)
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.raw_key = nn.Linear(hidden_size, memory_dim)
        self.address_key = nn.Linear(hidden_size, memory_dim)
        self.address_query = nn.Linear(hidden_size, memory_dim)
        self.raw_value = nn.Linear(hidden_size, memory_dim)
        self.self_proj = nn.Linear(hidden_size * 2, memory_dim)
        self.use_proj = nn.Linear(hidden_size, memory_dim)
        self.delta_q = nn.Linear(memory_dim, memory_dim)
        self.delta_k = nn.Linear(memory_dim, memory_dim)
        self.delta_v = nn.Linear(memory_dim, memory_dim)
        self.norm = RMSNorm(memory_dim, eps=eps)
        # Stage 6: token-preserving attention pooling. Learned query reads
        # value-span hidden states with an fp32 softmax (MPS/bf16 safe).
        self.attn_pool_query = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.attn_pool_query, std=hidden_size ** -0.5)

    def write_layer(
        self,
        *,
        layer_id: int,
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        attn: torch.Tensor | None,
        token_offset: int = 0,
        source_text_by_block: Sequence[str] | None = None,
    ) -> list[AttentionMemoryItem]:
        if h_in.dim() != 3 or h_out.dim() != 3:
            raise ValueError("h_in and h_out must have shape [batch, seq, hidden]")
        if h_in.shape[0] != 1:
            raise ValueError("P0 writer currently supports batch size 1")
        seq_len = min(h_in.shape[1], h_out.shape[1])
        attn_mean = None
        if attn is not None:
            if attn.dim() != 4:
                raise ValueError("attn must have shape [batch, heads, seq, seq]")
            attn_mean = attn[:, :, :seq_len, :seq_len].float().mean(dim=1)

        items: list[AttentionMemoryItem] = []
        for block_id, start in enumerate(range(0, seq_len, self.block_size)):
            end = min(start + self.block_size, seq_len)
            h_block_in = h_in[0, start:end]
            h_block_out = h_out[0, start:end]
            raw_summary = h_block_out.mean(dim=0)
            raw_key = self.raw_key(raw_summary)
            address_key = self.address_key(raw_summary)
            raw_value = self.raw_value(raw_summary)
            delta, usage_mass = self._delta_value(h_block_in, h_block_out, h_out[0], attn_mean, start, end)
            snippet = ""
            if source_text_by_block is not None and block_id < len(source_text_by_block):
                snippet = source_text_by_block[block_id]
            metadata = {
                "source_text": snippet,
                "source_text_debug_only": True,
                "layer_id": layer_id,
                "block_id": block_id,
                "token_range": [token_offset + start, token_offset + end],
                "usage_mass": usage_mass,
                "block_size": self.block_size,
            }
            items.append(
                AttentionMemoryItem(
                    memory_id=None,
                    layer_id=layer_id,
                    block_id=block_id,
                    token_start=token_offset + start,
                    token_end=token_offset + end,
                    raw_key=raw_key,
                    address_key=address_key,
                    raw_value=raw_value,
                    delta_q=self.delta_q(delta),
                    delta_k=self.delta_k(delta),
                    delta_v=self.delta_v(delta),
                    usage_mass=usage_mass,
                    metadata=metadata,
                )
            )
        return items

    def write_oracle_span_layer(
        self,
        *,
        layer_id: int,
        h_out: torch.Tensor,
        address_token_range: tuple[int, int],
        value_token_range: tuple[int, int],
        token_offset: int = 0,
        source_text: str = "",
        pool: str = "mean",
    ) -> list[AttentionMemoryItem]:
        if h_out.dim() != 3:
            raise ValueError("h_out must have shape [batch, seq, hidden]")
        if h_out.shape[0] != 1:
            raise ValueError("oracle span writer currently supports batch size 1")
        address_start, address_end = _clamped_span(address_token_range, h_out.shape[1])
        value_start, value_end = _clamped_span(value_token_range, h_out.shape[1])
        address_tokens = h_out[0, address_start:address_end]
        value_tokens = h_out[0, value_start:value_end]
        if pool == "attn":
            address_summary = self._attn_pool(address_tokens)
            value_summary = self._attn_pool(value_tokens)
        elif pool == "mean":
            address_summary = address_tokens.mean(dim=0)
            value_summary = value_tokens.mean(dim=0)
        else:
            raise ValueError(f"unsupported writer pool: {pool}")
        raw_key = self.raw_key(address_summary)
        address_key = self.address_key(address_summary)
        raw_value = self.raw_value(value_summary)
        payload_basis = self.self_proj(torch.cat([address_summary, value_summary], dim=-1))
        payload_use = self.use_proj(value_summary)
        delta = self.norm(payload_use - payload_basis)
        if not torch.isfinite(delta).all():
            delta = torch.nan_to_num(delta)
        metadata = {
            "source_text": source_text,
            "source_text_debug_only": True,
            "layer_id": layer_id,
            "block_id": 0,
            "token_range": [token_offset + address_start, token_offset + value_end],
            "address_token_range": [token_offset + address_start, token_offset + address_end],
            "value_token_range": [token_offset + value_start, token_offset + value_end],
            "usage_mass": 1.0,
            "block_size": self.block_size,
            "oracle_span_writer": True,
            "writer_pool": pool,
        }
        return [
            AttentionMemoryItem(
                memory_id=None,
                layer_id=layer_id,
                block_id=0,
                token_start=token_offset + address_start,
                token_end=token_offset + value_end,
                raw_key=raw_key,
                address_key=address_key,
                raw_value=raw_value,
                delta_q=self.delta_q(delta),
                delta_k=self.delta_k(delta),
                delta_v=self.delta_v(delta),
                usage_mass=1.0,
                metadata=metadata,
            )
        ]

    def _attn_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token-preserving attention pool over a span.

        ``tokens`` has shape ``[span_len, hidden]``. Computes softmax in fp32
        for MPS/bf16 numerical safety, then casts back to the input dtype.
        """
        if tokens.numel() == 0:
            return torch.zeros(self.hidden_size, device=tokens.device, dtype=tokens.dtype)
        if tokens.shape[0] == 1:
            return tokens[0]
        query = self.attn_pool_query.to(device=tokens.device, dtype=tokens.dtype)
        scores = (tokens.float() @ query.float()) / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=0).to(tokens.dtype)
        return (weights.unsqueeze(-1) * tokens).sum(dim=0)


    def _delta_value(
        self,
        h_block_in: torch.Tensor,
        h_block_out: torch.Tensor,
        h_out: torch.Tensor,
        attn_mean: torch.Tensor | None,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, float]:
        c_self_input = torch.cat([h_block_in.mean(dim=0), h_block_out.mean(dim=0)], dim=-1)
        c_self = self.self_proj(c_self_input)
        if attn_mean is None or end >= h_out.shape[0]:
            return torch.zeros_like(c_self), 0.0
        future = h_out[end:]
        c_use = self.use_proj(future)
        weights = attn_mean[0, end : h_out.shape[0], start:end].sum(dim=-1).float()
        usage_mass_tensor = weights.sum()
        if float(usage_mass_tensor.detach().cpu()) <= self.eps:
            return torch.zeros_like(c_self), 0.0
        weights = weights / (usage_mass_tensor + self.eps)
        v2 = (weights.unsqueeze(-1).to(c_use.dtype) * c_use).sum(dim=0)
        delta = self.norm(v2 - c_self)
        if not torch.isfinite(delta).all():
            delta = torch.nan_to_num(delta)
        return delta, float(usage_mass_tensor.detach().cpu())


def split_source_snippets(tokenizer, input_ids: torch.Tensor, block_size: int) -> list[str]:
    ids = input_ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    snippets = []
    for start in range(0, len(ids), block_size):
        chunk = ids[start : start + block_size]
        try:
            snippets.append(tokenizer.decode(chunk, skip_special_tokens=True))
        except TypeError:
            snippets.append(tokenizer.decode(chunk))
    return snippets


def fit_memory_dim(vector: torch.Tensor, memory_dim: int) -> torch.Tensor:
    flat = vector.flatten()
    if flat.numel() == memory_dim:
        return flat
    if flat.numel() > memory_dim:
        return flat[:memory_dim]
    return F.pad(flat, (0, memory_dim - flat.numel()))


def _clamped_span(span: tuple[int, int], seq_len: int) -> tuple[int, int]:
    start = max(0, min(int(span[0]), seq_len - 1))
    end = max(start + 1, min(int(span[1]), seq_len))
    return start, end
