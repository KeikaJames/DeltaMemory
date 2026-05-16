"""AttentionBank and learnable heads for LPL.

Bank stores per-layer (round_idx, position, h ∈ R^d) entries.  Read path
projects stored ``h`` through the *current* layer's W_K, W_V (after Qwen3's
k_norm) and concats the resulting K, V to the standard softmax — bank slice
skips RoPE (out-of-time semantic pool, sized to bank-only attention scores).

This module owns no Qwen3 specifics; the projection is delegated to the
attention patch in ``qwen3_lpl_patch.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class AttentionBank:
    """Per-layer dynamic working-memory.

    ``slots[l]`` is a tensor of shape ``[N_l, d]`` (lazy alloc) collecting
    h-vectors written when layer l paused at some position in some round.

    A simple FIFO cap (``max_per_layer``) bounds growth.
    """

    num_layers: int
    hidden_size: int
    max_per_layer: int = 256
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.bfloat16

    # populated lazily
    slots: list[torch.Tensor] = field(default_factory=list)
    # tag = (round_idx, position) per entry, for debugging / future use
    tags: list[list[tuple[int, int]]] = field(default_factory=list)

    # If True, freeze the bank (no further writes). Used by Phase D when
    # the bank has been preloaded from Exp35b/38 and should act as a pure
    # static long-term-memory pool.
    frozen: bool = False

    def __post_init__(self) -> None:
        if not self.slots:
            for _ in range(self.num_layers):
                self.slots.append(
                    torch.empty(0, self.hidden_size, device=self.device, dtype=self.dtype)
                )
                self.tags.append([])

    def is_empty(self, layer: int | None = None) -> bool:
        if layer is None:
            return all(t.shape[0] == 0 for t in self.slots)
        return self.slots[layer].shape[0] == 0

    def total_size(self) -> int:
        return sum(t.shape[0] for t in self.slots)

    def clear(self) -> None:
        """Drop all entries (call between independent inferences)."""
        for l in range(self.num_layers):
            self.slots[l] = torch.empty(
                0, self.hidden_size, device=self.device, dtype=self.dtype
            )
            self.tags[l] = []

    def write(
        self,
        layer: int,
        h_in_at_paused: torch.Tensor,
        positions: list[tuple[int, int]],  # (batch_idx, seq_pos) per row
        round_idx: int,
    ) -> None:
        """Append paused-position hidden vectors to layer ``layer``.

        Args:
            layer: which layer is writing.
            h_in_at_paused: tensor [N, d] of pre-norm hidden vectors at paused
                positions (one row per paused position, batch-flattened).
            positions: parallel list of (batch_idx, seq_pos) for tagging.
            round_idx: current round τ.
        """
        if self.frozen or h_in_at_paused.shape[0] == 0:
            return
        new = h_in_at_paused.detach().to(device=self.device, dtype=self.dtype)
        current = self.slots[layer]
        merged = torch.cat([current, new], dim=0)
        if merged.shape[0] > self.max_per_layer:
            # FIFO: drop oldest
            keep = merged.shape[0] - self.max_per_layer
            merged = merged[keep:]
            self.tags[layer] = self.tags[layer][keep:]
        self.slots[layer] = merged
        self.tags[layer].extend((round_idx, pos) for (_, pos) in positions)

    def read(self, layer: int) -> torch.Tensor | None:
        """Return [N_l, d] tensor of bank h-vectors for ``layer``, or None if empty."""
        t = self.slots[layer]
        if t.shape[0] == 0:
            return None
        return t


class PauseHead(nn.Module):
    """Per-layer single-linear pause head: h -> p_pause in [0,1].

    Initialized with bias=-10.0 so σ(W·h + b) ≈ 0 (no pause anywhere) until
    trained. This guarantees Gate 0 bit-equal baseline.
    """

    def __init__(self, hidden_size: int, init_bias: float = -10.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., d] -> [..., 1] in (0,1)."""
        return torch.sigmoid(self.proj(h.float()))


class BankGateHead(nn.Module):
    """Per-layer per-position scalar multiplier on bank V (replaces v1 global α).

    Initialized so σ→1.0 (init_bias=10.0): when bank is non-empty its V is
    used at full strength by default. Trainable.
    """

    def __init__(self, hidden_size: int, init_bias: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., d] -> [..., 1] in (0,1)."""
        return torch.sigmoid(self.proj(h.float()))


class HaltHead(nn.Module):
    """Single halt head: last-layer last-token hidden -> p_halt in [0,1].

    Initialized with bias=+10.0 → σ ≈ 1.0 so we halt after K=1 by default
    (Gate 0). Trainable in Phase C.
    """

    def __init__(self, hidden_size: int, init_bias: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h.float()))


@dataclass
class LPLHeads:
    """Container for the trainable parameters of LPL."""

    pause_heads: nn.ModuleList   # ModuleList of PauseHead, one per layer
    bank_gate_heads: nn.ModuleList  # ModuleList of BankGateHead, one per layer
    halt_head: HaltHead

    @classmethod
    def fresh(
        cls,
        num_layers: int,
        hidden_size: int,
        *,
        pause_bias: float = -10.0,
        bank_gate_bias: float = 10.0,
        halt_bias: float = 10.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "LPLHeads":
        pause = nn.ModuleList(
            [PauseHead(hidden_size, init_bias=pause_bias) for _ in range(num_layers)]
        )
        gate = nn.ModuleList(
            [BankGateHead(hidden_size, init_bias=bank_gate_bias) for _ in range(num_layers)]
        )
        halt = HaltHead(hidden_size, init_bias=halt_bias)
        # Heads stay in fp32 for stable gradients; move to device.
        pause = pause.to(device=device, dtype=dtype)
        gate = gate.to(device=device, dtype=dtype)
        halt = halt.to(device=device, dtype=dtype)
        return cls(pause_heads=pause, bank_gate_heads=gate, halt_head=halt)

    def parameters(self):
        for p in self.pause_heads.parameters():
            yield p
        for p in self.bank_gate_heads.parameters():
            yield p
        for p in self.halt_head.parameters():
            yield p
