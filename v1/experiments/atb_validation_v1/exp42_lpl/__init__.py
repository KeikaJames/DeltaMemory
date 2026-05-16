"""Exp42 — Latent Pause Loop (LPL) with AttentionBank.

See plan.md (Exp42 LPL) for design. This package implements:
- AttentionBank: per-layer dynamic store of hidden vectors written when a
  layer "pauses" at a position (residual-skip).
- Qwen3 attention/decoder-layer patches that (a) honor per-position pause
  and (b) concat bank-derived K/V into the attention softmax on later
  rounds — direct generalization of v1 AttnNativeBank's
  ``Attn(Q,[K;M_K],[V;α·M_V])`` to dynamic, learnable, multi-round.
- LPLRuntime: multi-round forward loop with optional ACT halt head.

Gate 0 sanity: when bank empty + all pause heads disabled + K_max=1, the
patched model is bit-equal to the base model.
"""

from .attention_bank import AttentionBank, LPLHeads
from .qwen3_lpl_patch import install_lpl_patch, uninstall_lpl_patch
from .runtime import LPLRuntime, LPLConfig

__all__ = [
    "AttentionBank",
    "LPLHeads",
    "install_lpl_patch",
    "uninstall_lpl_patch",
    "LPLRuntime",
    "LPLConfig",
]
