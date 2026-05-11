"""Residual Stream Memory (RSM).

RSM stores one residual-stream vector per decoder layer from a write prompt,
then replays selected memories into the read prompt's residual stream.  Read
selection is two-pass by design: first probe the read prompt without mutation to
score memories by max-layer cosine similarity, then run a second forward with a
fixed score vector injected at every layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from deltamemory.memory._layer_locator import get_decoder_layers

RSMHookPoint = Literal["block_output", "pre_block_input", "mlp_mid"]
RSMGateMode = Literal["threshold", "off", "uniform"]


@dataclass
class RSMConfig:
    """RSM read/write hyperparameters."""

    eta: float = 0.1
    theta: float = 0.5
    hook_point: RSMHookPoint = "block_output"
    # Gate semantics:
    #   "threshold" (default): weights = where(scores > theta, scores, 0).
    #   "off": skip theta threshold but keep nonnegative cosine weights
    #          (weights = clamp_min(scores, 0)).  Still selectivity-weighted.
    #   "uniform": ignore both theta and score magnitude — every bank entry
    #              contributes with weight 1.  Pure "global steering" stress.
    gate_mode: RSMGateMode = "threshold"
    # Back-compat shim: if gate_off=True is supplied at construction time we
    # promote it to gate_mode="off" in __post_init__.
    gate_off: bool = False
    inject_only_last_token: bool = True

    def __post_init__(self) -> None:
        if self.gate_off and self.gate_mode == "threshold":
            self.gate_mode = "off"


@dataclass
class RSMMemoryBank:
    """A residual-memory bank with tensor shape ``(N, L, D)``."""

    memories: torch.Tensor
    fact_ids: list[str]

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("RSMMemoryBank.memories must have shape (N, L, D).")
        if self.memories.shape[0] != len(self.fact_ids):
            raise ValueError("fact_ids length must match memories.shape[0].")
        self.memories = self.memories.detach().cpu().float().contiguous()
        self.fact_ids = [str(fid) for fid in self.fact_ids]

    @property
    def n_memories(self) -> int:
        return int(self.memories.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.memories.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.memories.shape[2])

    def shuffled_layers(self, *, seed: int = 0xA11CE) -> "RSMMemoryBank":
        """Return a copy with the layer axis permuted."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(self.n_layers, generator=gen)
        return RSMMemoryBank(
            memories=self.memories[:, perm, :].contiguous(),
            fact_ids=list(self.fact_ids),
        )


def _last_index(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> int:
    if input_ids.shape[0] != 1:
        raise ValueError("RSM currently supports batch_size=1.")
    if attention_mask is not None:
        return max(0, int(attention_mask[0].sum().item()) - 1)
    return max(0, int(input_ids.shape[1]) - 1)


def _as_hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def _resolve_target_module(layer: Any, hook_point: RSMHookPoint) -> Any:
    """Return the nn.Module on which to attach the hook for ``hook_point``."""
    if hook_point in ("block_output", "pre_block_input"):
        return layer
    if hook_point == "mlp_mid":
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("RSM mlp_mid hook requires a layer with a .mlp child.")
        down = getattr(mlp, "down_proj", None)
        if down is None:
            raise ValueError(
                "RSM mlp_mid hook requires .mlp.down_proj (SwiGLU-style). "
                f"Found mlp children: {[n for n, _ in mlp.named_children()]}"
            )
        return down
    raise ValueError(f"Unsupported RSM hook_point: {hook_point}")


class RSMInjector:
    """Hook-based RSM capture and two-pass read injector."""

    def __init__(self, model: Any, config: RSMConfig | None = None) -> None:
        self.model = model
        self.config = config or RSMConfig()
        self.layers = get_decoder_layers(model)
        # Validate hook_point early so a bad config fails before forward calls.
        for layer in self.layers:
            _resolve_target_module(layer, self.config.hook_point)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hook_point(self) -> RSMHookPoint:
        return self.config.hook_point

    def _target_modules(self) -> list[Any]:
        return [_resolve_target_module(layer, self.config.hook_point) for layer in self.layers]

    def _is_pre_hook(self) -> bool:
        return self.config.hook_point in ("pre_block_input", "mlp_mid")

    @torch.no_grad()
    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture last-token activation vectors, returning ``(L, D)`` CPU fp32.

        ``D`` follows the hook point:
        - ``block_output`` / ``pre_block_input``: ``hidden_size``.
        - ``mlp_mid``: ``intermediate_size`` (input to ``mlp.down_proj``).
        """
        last_idx = _last_index(input_ids, attention_mask)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        use_pre = self._is_pre_hook()
        targets = self._target_modules()

        def make_post_hook(layer_idx: int):
            def _hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                hidden = _as_hidden(output)
                captured[layer_idx] = hidden[0, last_idx, :].detach().cpu().float()
            return _hook

        def make_pre_hook(layer_idx: int):
            def _hook(module: Any, inputs: tuple[Any, ...]) -> None:
                hidden = inputs[0]
                captured[layer_idx] = hidden[0, last_idx, :].detach().cpu().float()
            return _hook

        for layer_idx, mod in enumerate(targets):
            if use_pre:
                handles.append(mod.register_forward_pre_hook(make_pre_hook(layer_idx)))
            else:
                handles.append(mod.register_forward_hook(make_post_hook(layer_idx)))

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self.model.train()

        missing = [idx for idx in range(self.num_layers) if idx not in captured]
        if missing:
            raise RuntimeError(f"RSM capture missed layers: {missing}")
        return torch.stack([captured[idx] for idx in range(self.num_layers)], dim=0)

    def capture_text(
        self,
        tokenizer: Any,
        text: str,
        *,
        device: torch.device | str | None = None,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Tokenize ``text`` and capture an ``(L, D)`` residual memory."""
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return self.capture(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def score(
        self,
        bank: RSMMemoryBank,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-memory max-layer cosine scores, shape ``(N,)``."""
        if bank.n_memories == 0:
            return torch.empty(0, dtype=torch.float32)
        query = self.capture(input_ids=input_ids, attention_mask=attention_mask)
        if query.shape[0] != bank.n_layers or query.shape[1] != bank.hidden_dim:
            raise ValueError(
                "query memory shape does not match bank: "
                f"query={tuple(query.shape)} bank={tuple(bank.memories.shape)}"
            )
        q = F.normalize(query.float(), dim=-1)
        m = F.normalize(bank.memories.float(), dim=-1)
        sims = torch.einsum("ld,nld->nl", q, m)
        return sims.max(dim=1).values

    @torch.no_grad()
    def forward_with_memory(
        self,
        bank: RSMMemoryBank,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Run a two-pass RSM forward and return ``(model_output, diagnostics)``."""
        scores = self.score(bank, input_ids=input_ids, attention_mask=attention_mask)
        return self.forward_with_scores(
            bank,
            scores,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **forward_kwargs,
        )

    @torch.no_grad()
    def forward_with_scores(
        self,
        bank: RSMMemoryBank,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **forward_kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Run an RSM forward with caller-supplied per-memory scores.

        This is the leakage-safe path for teacher-forced logprob evaluation:
        callers can compute scores on the prompt-only prefix, then reuse those
        fixed scores while scoring target-token prefixes.
        """
        scores = scores.detach().cpu().float()
        if scores.ndim != 1 or scores.numel() != bank.n_memories:
            raise ValueError(
                "scores must have shape (N,) matching bank memories; "
                f"scores={tuple(scores.shape)} bank_n={bank.n_memories}"
            )
        gate_mode = self.config.gate_mode
        if gate_mode == "threshold":
            active = scores > float(self.config.theta)
            weights = torch.where(active, scores, torch.zeros_like(scores))
        elif gate_mode == "off":
            weights = scores.clamp_min(0.0)
            active = weights > 0.0
        elif gate_mode == "uniform":
            active = torch.ones_like(scores, dtype=torch.bool)
            weights = torch.ones_like(scores)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"Unsupported RSM gate_mode: {gate_mode}")

        if scores.numel():
            score_mean = float(scores.mean().item())
            score_min = float(scores.min().item())
            score_max = float(scores.max().item())
            score_std = float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0
            top_idx = int(scores.argmax().item())
            top_score = float(scores[top_idx].item())
            top_minus_mean = top_score - score_mean
            top_fact = bank.fact_ids[top_idx]
        else:
            score_mean = score_min = score_max = score_std = float("nan")
            top_idx = -1
            top_score = float("nan")
            top_minus_mean = float("nan")
            top_fact = None

        diag = {
            "rsm_scores": scores.detach().cpu(),
            "rsm_active": active.detach().cpu(),
            "rsm_activation_rate": float(active.float().mean().item()) if active.numel() else 0.0,
            "rsm_max_score": score_max,
            "rsm_min_score": score_min,
            "rsm_mean_score": score_mean,
            "rsm_score_std": score_std,
            "rsm_top_score_minus_mean": top_minus_mean,
            "rsm_top_index": top_idx,
            "rsm_top_fact_id": top_fact,
            "rsm_hook_point": self.config.hook_point,
            "rsm_gate_mode": gate_mode,
        }

        if float(self.config.eta) == 0.0 or bank.n_memories == 0 or float(weights.abs().sum().item()) == 0.0:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **forward_kwargs,
            )
            return out, diag

        last_idx = _last_index(input_ids, attention_mask)
        memories = bank.memories.to(device=input_ids.device, dtype=torch.float32)
        weights_dev = weights.to(device=input_ids.device, dtype=torch.float32)
        handles = []
        use_pre = self._is_pre_hook()
        targets = self._target_modules()

        def _apply_delta(hidden: torch.Tensor, delta_local: torch.Tensor) -> torch.Tensor:
            if self.config.inject_only_last_token:
                new_hidden = hidden.clone()
                new_hidden[0, last_idx, :] = new_hidden[0, last_idx, :] + delta_local
                return new_hidden
            return hidden + delta_local.view(1, 1, -1)

        def make_post_hook(layer_idx: int):
            delta = torch.einsum("n,nd->d", weights_dev, memories[:, layer_idx, :])

            def _hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = _as_hidden(output)
                delta_local = (float(self.config.eta) * delta).to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                return _replace_hidden(output, _apply_delta(hidden, delta_local))

            return _hook

        def make_pre_hook(layer_idx: int):
            delta = torch.einsum("n,nd->d", weights_dev, memories[:, layer_idx, :])

            def _hook(module: Any, inputs: tuple[Any, ...]) -> Any:
                hidden = inputs[0]
                delta_local = (float(self.config.eta) * delta).to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                new_hidden = _apply_delta(hidden, delta_local)
                return (new_hidden,) + tuple(inputs[1:])

            return _hook

        for layer_idx, mod in enumerate(targets):
            if use_pre:
                handles.append(mod.register_forward_pre_hook(make_pre_hook(layer_idx)))
            else:
                handles.append(mod.register_forward_hook(make_post_hook(layer_idx)))

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **forward_kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self.model.train()
        return out, diag


__all__ = ["RSMConfig", "RSMInjector", "RSMMemoryBank", "RSMHookPoint", "RSMGateMode"]
