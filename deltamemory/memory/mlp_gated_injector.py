"""Exp32 — MLP-side gated memory injector.

Tests hypothesis H_B: ANB failure on attention is a *site* problem, not a
*routing* problem.  Injects a separate (M_K, M_V) bank at the **MLP output**
of each decoder layer, with **learned** per-layer routing and gating.

Per-layer forward:

    score_l   = W_q_l(h_in) @ M_K_l^T               # (N,)
    weights_l = softmax(score_l / tau)              # or topk
    v_l       = weights_l @ M_V_l                   # (d_model,)
    gate_l    = sigmoid(W_g_l(h_in))                # scalar
    h_out     = MLP_l(h_in) + gate_l * v_l          # @ last token only

M_K_l and M_V_l are captured from a write forward pass — by default both at
the relation_last token position, post-MLP residual stream.  Base model is
fully frozen; only (W_q, W_g) are trained.

This module is independent of the attention-side ATB and can be tested in
isolation.  See `experiments/atb_validation_v1/exp32_mlp_side_gated_memory/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deltamemory.memory._layer_locator import get_decoder_layers


@dataclass
class MLPGatedConfig:
    key_dim: int = 64
    temperature: float = 0.07
    topk: int = 0          # 0 = full softmax; >0 = sparse topk
    inject_only_last_token: bool = True
    eta: float = 1.0        # outer scale on top of learned gate
    gate_mode: str = "learned"   # "learned" | "fixed_one" | "off"


class MLPGatedRouter(nn.Module):
    """Per-layer learned (W_q, W_g). Identity-safe init (gate ~ 0)."""

    def __init__(self, num_layers: int, d_model: int, key_dim: int = 64) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.key_dim = key_dim
        self.W_q = nn.ModuleList([nn.Linear(d_model, key_dim, bias=False)
                                  for _ in range(num_layers)])
        self.W_g = nn.ModuleList([nn.Linear(d_model, 1, bias=True)
                                  for _ in range(num_layers)])
        # Init: gate bias = -4 ⇒ sigmoid ~ 0.018 ⇒ bank ~ off at init.
        # This guarantees identity-equivalence to base when bank is empty
        # and lets training discover when to open the gate.
        for g in self.W_g:
            nn.init.constant_(g.bias, -4.0)
            nn.init.zeros_(g.weight)
        for q in self.W_q:
            nn.init.xavier_uniform_(q.weight)


@dataclass
class MLPMemoryBank:
    """(M_K, M_V): per-layer key and value tensors. Lazy device move."""
    M_K: list[torch.Tensor]       # each: (N, d_model)
    M_V: list[torch.Tensor]       # each: (N, d_model)
    fact_ids: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.M_K)

    @property
    def n_facts(self) -> int:
        return self.M_K[0].shape[0] if self.M_K else 0

    def to(self, device: Any, dtype: Any) -> "MLPMemoryBank":
        return MLPMemoryBank(
            M_K=[k.to(device=device, dtype=dtype) for k in self.M_K],
            M_V=[v.to(device=device, dtype=dtype) for v in self.M_V],
            fact_ids=list(self.fact_ids),
        )


class MLPGatedInjector:
    """Installs forward hooks on each decoder layer's `.mlp` module.

    `inject(bank, router)` returns a context manager that activates the hooks;
    `capture(...)` records (h_pre_mlp, h_post_mlp) at a specified token position
    for bank construction.
    """

    def __init__(self, model: Any, config: MLPGatedConfig | None = None) -> None:
        self.model = model
        self.cfg = config or MLPGatedConfig()
        self.layers = get_decoder_layers(model)
        self.num_layers = len(self.layers)
        self._handles: list[Any] = []
        self._capture_targets: dict[int, dict[str, torch.Tensor]] = {}

    # --- capture path --------------------------------------------------

    def capture_at_pos(
        self,
        input_ids: torch.Tensor,
        pos: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture per-layer (h_pre_mlp[pos], h_post_mlp[pos]).

        Returns (K_tensor, V_tensor), each shape (L, d_model) on CPU fp32.

        K = h_in (input to MLP) at position `pos` — the routing query/key
        V = MLP output at position `pos` — the memory value
        """
        captured: dict[int, dict[str, torch.Tensor]] = {l: {} for l in range(self.num_layers)}
        handles: list[Any] = []

        def make_pre(l: int):
            def _h(mod: Any, inputs: tuple[Any, ...]) -> None:
                h = inputs[0]
                captured[l]["pre"] = h[0, pos, :].detach().cpu().float()
            return _h

        def make_post(l: int):
            def _h(mod: Any, inputs: tuple[Any, ...], output: Any) -> None:
                h = output[0] if isinstance(output, tuple) else output
                captured[l]["post"] = h[0, pos, :].detach().cpu().float()
            return _h

        for li, layer in enumerate(self.layers):
            mlp = layer.mlp
            handles.append(mlp.register_forward_pre_hook(make_pre(li)))
            handles.append(mlp.register_forward_hook(make_post(li)))

        try:
            self.model.eval()
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        K = torch.stack([captured[l]["pre"] for l in range(self.num_layers)], dim=0)
        V = torch.stack([captured[l]["post"] for l in range(self.num_layers)], dim=0)
        return K, V

    # --- inject path ---------------------------------------------------

    def install(self, bank: MLPMemoryBank, router: MLPGatedRouter,
                last_token_idx: int) -> None:
        """Install forward hooks that read the bank into MLP output."""
        if bank.n_layers != self.num_layers:
            raise ValueError(f"bank.n_layers={bank.n_layers} != model.L={self.num_layers}")
        self.remove()
        cfg = self.cfg

        for li, layer in enumerate(self.layers):
            mlp = layer.mlp
            MK_l = bank.M_K[li]
            MV_l = bank.M_V[li]
            Wq_l = router.W_q[li]
            Wg_l = router.W_g[li]
            key_dim = router.key_dim

            # Pre-project MK to key_dim — done lazily per fwd (router may train)
            def make_hook(MK_l=MK_l, MV_l=MV_l, Wq_l=Wq_l, Wg_l=Wg_l, key_dim=key_dim):
                def _h(mod: Any, inputs: tuple[Any, ...], output: Any) -> Any:
                    h_in = inputs[0]
                    h_out = output[0] if isinstance(output, tuple) else output
                    if MV_l.shape[0] == 0:
                        return output
                    device, dtype = h_out.device, h_out.dtype
                    h_query = h_in[0, last_token_idx, :].to(dtype=torch.float32)
                    # Project query to key_dim; project MK to key_dim
                    q = Wq_l(h_query)                                # (key_dim,)
                    mk = Wq_l(MK_l.to(device=device, dtype=torch.float32))  # (N, key_dim)
                    score = (q @ mk.T) / cfg.temperature             # (N,)
                    if cfg.topk and cfg.topk < mk.shape[0]:
                        topv, topi = score.topk(cfg.topk)
                        w = torch.zeros_like(score)
                        w[topi] = F.softmax(topv, dim=-1)
                    else:
                        w = F.softmax(score, dim=-1)
                    v = w @ MV_l.to(device=device, dtype=torch.float32)   # (d_model,)
                    if cfg.gate_mode == "learned":
                        gate = torch.sigmoid(Wg_l(h_query)).squeeze(-1)
                    elif cfg.gate_mode == "fixed_one":
                        gate = torch.ones((), device=device, dtype=torch.float32)
                    else:  # off
                        gate = torch.zeros((), device=device, dtype=torch.float32)
                    delta = (cfg.eta * gate * v).to(dtype=dtype)
                    new_out = h_out.clone()
                    if cfg.inject_only_last_token:
                        new_out[0, last_token_idx, :] = new_out[0, last_token_idx, :] + delta
                    else:
                        new_out = new_out + delta.view(1, 1, -1)
                    return (new_out,) + output[1:] if isinstance(output, tuple) else new_out
                return _h

            self._handles.append(mlp.register_forward_hook(make_hook()))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
