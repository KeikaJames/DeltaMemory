"""LPLRuntime — multi-round forward + ACT halt orchestration.

For Phase 0/Gate 0 the runtime is intentionally simple: it just runs ``K_max``
rounds of full forward, threading the AttentionBank across rounds and letting
the patched layers do the pause/skip/bank-augment work.

In Phase C this will gain ACT halt accumulation and early-exit. For now,
``halt_use=True`` simply checks p_halt and stops when cumulative halt > eps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .attention_bank import AttentionBank, LPLHeads
from .qwen3_lpl_patch import LPLState, install_lpl_patch, lpl_state_scope


@dataclass
class LPLConfig:
    K_max: int = 1
    halt_eps: float = 0.05
    enabled: bool = True
    use_halt_head: bool = False  # Phase C will set True
    force_pause_mask: Optional[torch.Tensor] = None  # Phase A frozen handcraft


@dataclass
class LPLForwardResult:
    logits: torch.Tensor                    # [B, T, V] from final round
    rounds_used: int
    per_round_logits: list[torch.Tensor] = field(default_factory=list)
    pause_count_per_layer: list[int] = field(default_factory=list)
    bank_total_size_after: int = 0


class LPLRuntime:
    def __init__(self, model, heads: LPLHeads | None, bank: AttentionBank, config: LPLConfig):
        self.model = model
        self.heads = heads
        self.bank = bank
        self.config = config
        if not getattr(model, "_lpl_patched", False):
            install_lpl_patch(model, state=None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                clear_bank: bool = True) -> LPLForwardResult:
        if clear_bank:
            self.bank.clear()
        per_round_logits = []
        last_logits = None
        rounds_used = 0
        cumulative_keep = 1.0

        for tau in range(1, self.config.K_max + 1):
            state = LPLState(
                bank=self.bank,
                heads=self.heads,
                round_idx=tau,
                enabled=self.config.enabled,
                force_pause_mask=self.config.force_pause_mask,
            )
            with lpl_state_scope(self.model, state):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=False, return_dict=True)
            logits = out.logits  # [B, T, V]
            per_round_logits.append(logits)
            last_logits = logits
            rounds_used = tau

            if self.config.use_halt_head and self.heads is not None and tau < self.config.K_max:
                last_h = state.last_layer_hidden  # [B, T, d]
                # halt on last position
                p_halt = self.heads.halt_head(last_h[:, -1, :])  # [B, 1]
                cumulative_keep *= float((1.0 - p_halt).mean().item())
                if cumulative_keep < self.config.halt_eps:
                    break

        return LPLForwardResult(
            logits=last_logits,
            rounds_used=rounds_used,
            per_round_logits=per_round_logits,
            pause_count_per_layer=list(state.pause_count_per_layer),
            bank_total_size_after=self.bank.total_size(),
        )
