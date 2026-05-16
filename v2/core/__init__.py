"""v2 core: ALB (Attention-Side Latent Bank) building blocks."""
from .attention_bank import AttentionBank, LPLHeads, PauseHead, BankGateHead, HaltHead
from .qwen3_lpl_patch import install_lpl_patch, uninstall_lpl_patch, LPLState, lpl_state_scope
from .runtime import LPLRuntime, LPLConfig
from .kproj import LowRankProj, FullProj, make_projector, residual_apply
from .load_model import load_model
from .eval_lib import nll_on_answer, encode_qa, mean_nll
from . import data_io, retrieval, interrupt_api

__all__ = [
    "AttentionBank", "LPLHeads", "PauseHead", "BankGateHead", "HaltHead",
    "install_lpl_patch", "uninstall_lpl_patch", "LPLState", "lpl_state_scope",
    "LPLRuntime", "LPLConfig",
    "LowRankProj", "FullProj", "make_projector", "residual_apply",
    "load_model",
    "nll_on_answer", "encode_qa", "mean_nll",
    "data_io", "retrieval", "interrupt_api",
]
