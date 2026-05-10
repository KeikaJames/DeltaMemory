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

RSMHookPoint = Literal["block_output"]


@dataclass
class RSMConfig:
    """RSM read/write hyperparameters."""

    eta: float = 0.1
    theta: float = 0.5
    hook_point: RSMHookPoint = "block_output"
    # Skip the theta threshold but keep similarity weighting.  This is not an
    # all-ones stress path: weights are max(scores, 0).
    gate_off: bool = False
    inject_only_last_token: bool = True


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


class RSMInjector:
    """Hook-based RSM capture and two-pass read injector."""

    def __init__(self, model: Any, config: RSMConfig | None = None) -> None:
        self.model = model
        self.config = config or RSMConfig()
        if self.config.hook_point != "block_output":
            raise ValueError("RSMInjector currently supports hook_point='block_output'.")
        self.layers = get_decoder_layers(model)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @torch.no_grad()
    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture last-token residual vectors, returning ``(L, D)`` CPU fp32."""
        last_idx = _last_index(input_ids, attention_mask)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def _hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                hidden = _as_hidden(output)
                captured[layer_idx] = hidden[0, last_idx, :].detach().cpu().float()

            return _hook

        for layer_idx, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_hook(layer_idx)))

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
        if self.config.gate_off:
            weights = scores.clamp_min(0.0)
            active = weights > 0.0
        else:
            active = scores > float(self.config.theta)
            weights = torch.where(active, scores, torch.zeros_like(scores))

        diag = {
            "rsm_scores": scores.detach().cpu(),
            "rsm_active": active.detach().cpu(),
            "rsm_activation_rate": float(active.float().mean().item()) if active.numel() else 0.0,
            "rsm_max_score": float(scores.max().item()) if scores.numel() else float("nan"),
            "rsm_top_index": int(scores.argmax().item()) if scores.numel() else -1,
            "rsm_top_fact_id": bank.fact_ids[int(scores.argmax().item())] if scores.numel() else None,
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

        def make_hook(layer_idx: int):
            delta = torch.einsum("n,nd->d", weights_dev, memories[:, layer_idx, :])

            def _hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = _as_hidden(output)
                delta_local = (float(self.config.eta) * delta).to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                if self.config.inject_only_last_token:
                    new_hidden = hidden.clone()
                    new_hidden[0, last_idx, :] = new_hidden[0, last_idx, :] + delta_local
                else:
                    new_hidden = hidden + delta_local.view(1, 1, -1)
                return _replace_hidden(output, new_hidden)

            return _hook

        for layer_idx, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_hook(layer_idx)))

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


__all__ = ["RSMConfig", "RSMMemoryBank", "RSMInjector"]
