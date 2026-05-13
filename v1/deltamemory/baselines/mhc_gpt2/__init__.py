"""Vendored mHC (Manifold-Constrained Hyper-Connections) baseline.

Source: https://github.com/MarcoDotIO/mhc-deepseek-implementation (MIT)
Paper: arXiv:2512.24880 — DeepSeek "Manifold-Constrained Hyper-Connections"

See LICENSE and NOTICE in this directory for attribution.

This module is a *baseline* used by Phase mHC of Mneme; it is NOT part of
the Mneme main path. The Mneme red lines (frozen LLM weights) still
apply: when wired through `AttnNativePatcher`, only the bank-injection sites are
modified; the GPT-2 / mHC-GPT-2 weights themselves remain frozen at inference.
"""

from .mhc import MhcProjector
from .sinkhorn import sinkhorn_knopp
from .transformers.convert_gpt2 import convert_gpt2_lm_head_model
from .transformers.gpt2_mhc import MhcGPT2Config, MhcGPT2LMHeadModel, MhcGPT2Model

__all__ = [
    "MhcProjector",
    "sinkhorn_knopp",
    "MhcGPT2Config",
    "MhcGPT2Model",
    "MhcGPT2LMHeadModel",
    "convert_gpt2_lm_head_model",
]
