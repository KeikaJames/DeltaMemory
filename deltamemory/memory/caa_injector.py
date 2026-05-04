"""Phase X.3 — CAA-style residual-stream activation injector.

Contrastive Activation Addition (CAA; Rimsky+ 2024) injects a precomputed
steering vector ``s`` into the residual stream at a chosen transformer layer ℓ:

    h_ℓ' = h_ℓ + α · s

where ``s = mean_pos(h_ℓ) - mean_neg(h_ℓ)`` is built from a small contrastive
calibration set (positive vs negative prompt pairs).

This module is a *candidate replacement* for LOPI (W.4 experiment plan).  It is
zero-training (forward-only), architecture-agnostic via PyTorch hooks, and
inherits the same α=0 bit-equal red-line guarantee as LOPI: when α=0 no tensor
mutation occurs and the hook is a mathematical no-op.

Optional LOPI-style gating
--------------------------
When ``CAAConfig.use_lopi_gate = True`` the injection strength is modulated by
a per-token gate γ_t derived from the same Δ_Q derivative signal LOPI uses:

    h_ℓ' = h_ℓ + α · γ_t · s

``γ_t = sigmoid(k * (‖Q_t − Q_{t-1}‖₂ − θ))`` where Q is read from the first
element of the layer's *input* tuple (the hidden state).  This is an
approximation — the true Q requires an attention sub-module hook — but it
provides a free, self-consistent gate from the same signal family as LOPI.

Author: KeikaJames, 2026-05-04 (Phase X.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CAAConfig:
    """Hyperparameters for the CAA residual-stream injector.

    Attributes
    ----------
    inject_layer:
        Which transformer block to inject into.  Integer index (0-based) or
        the special sentinel ``"mu_arch"`` which resolves at construction time
        via :func:`~deltamemory.memory.lopi_profiler.profile_residuals` to
        the architecture's peak-variance layer.
    alpha:
        Global injection scale.  ``alpha=0`` ⇒ bit-equal identity (red-line).
    use_lopi_gate:
        If ``True``, multiply each injected vector by a per-token derivative
        gate γ_t (same family as LOPI's derivative gate).  Default ``False``
        for a clean baseline.
    gate_k:
        Sigmoid steepness used when ``use_lopi_gate=True``.
    gate_theta:
        Sigmoid threshold (shift) used when ``use_lopi_gate=True``.
    """

    inject_layer: int | Literal["mu_arch"] = "mu_arch"
    alpha: float = 1.0
    use_lopi_gate: bool = False   # default OFF — clean baseline
    gate_k: float = 5.0
    gate_theta: float = 0.5


# ---------------------------------------------------------------------------
# CAAInjector
# ---------------------------------------------------------------------------


class CAAInjector:
    """Residual-stream activation injector following the CAA / ActAdd paradigm.

    Usage::

        cfg   = CAAConfig(inject_layer="mu_arch", alpha=1.0)
        inj   = CAAInjector(model, cfg, tokenizer=tokenizer)
        s     = inj.calibrate(pos_texts, neg_texts)

        with inj:
            out = model(input_ids)          # injection active
        out = model(input_ids)              # injection removed

    The context-manager installs a single :meth:`torch.nn.Module.register_forward_hook`
    on ``model.model.layers[ℓ]`` (GPT-2 / Llama / Gemma path).  The hook adds
    ``α · s`` (optionally gated by γ_t) to the hidden-state tensor and returns
    the modified output tuple without touching any weight tensors.

    Parameters
    ----------
    model:
        HuggingFace causal-LM.  Weights are never mutated.
    config:
        :class:`CAAConfig` dataclass.
    tokenizer:
        Optional tokenizer required for :meth:`calibrate`.  May be supplied
        later via ``inj.tokenizer = tok`` before calling ``calibrate``.
    device:
        Inference device.  Defaults to the model's first-parameter device.
    """

    def __init__(
        self,
        model: Any,
        config: CAAConfig,
        tokenizer: Any = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._model = model
        self.config = config
        self.tokenizer = tokenizer

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self._device = device

        # Resolved integer layer index (None until _resolve_layer is called).
        self._inject_layer: Optional[int] = None

        # Steering vector — set by calibrate() or assign directly.
        self.steering_vector: Optional[torch.Tensor] = None

        # Active hook handle (non-None only inside the context manager).
        self._hook_handle: Optional[Any] = None

        # Previous hidden-state for the LOPI-style derivative gate.
        self._prev_hidden: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------

    def _resolve_layer(self) -> int:
        """Return the concrete (int) injection layer index.

        Resolves ``"mu_arch"`` lazily using the LOPI profiler and caches the
        result so subsequent calls are O(1).  Falls back to ``L // 2`` if the
        profiler cannot run (e.g. tokenizer not available).
        """
        if self._inject_layer is not None:
            return self._inject_layer

        if isinstance(self.config.inject_layer, int):
            self._inject_layer = self.config.inject_layer
            return self._inject_layer

        # "mu_arch" sentinel — try profiler first, fall back to L//2.
        layers = self._get_decoder_layers()
        L = len(layers)
        fallback = L // 2

        if self.tokenizer is not None:
            try:
                from deltamemory.memory.lopi_profiler import profile_residuals
                profile = profile_residuals(
                    self._model,
                    self.tokenizer,
                    device=self._device,
                )
                self._inject_layer = int(profile.mu_arch)
            except Exception:
                self._inject_layer = fallback
        else:
            self._inject_layer = fallback

        return self._inject_layer

    # ------------------------------------------------------------------
    # Decoder-layer discovery  (mirrors DiagnosticRecorder paths)
    # ------------------------------------------------------------------

    def _get_decoder_layers(self) -> list:
        model = self._model
        for path in (
            "model.model.language_model.layers",
            "model.model.layers",
            "model.language_model.model.layers",
            "model.language_model.layers",
            "language_model.layers",
            "model.layers",
            "transformer.h",   # GPT-2
        ):
            obj: Any = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        raise RuntimeError(
            "CAAInjector: could not locate decoder layers on the model. "
            "Ensure the model has a standard HuggingFace decoder-layer structure."
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        *,
        max_length: int = 64,
    ) -> torch.Tensor:
        """Compute steering vector ``s`` from contrastive prompt pairs.

        ``s = mean_pos(h_ℓ) - mean_neg(h_ℓ)``

        The mean is taken over all (batch, token) positions and then over the
        two lists, matching the CAA paper's formulation.

        Parameters
        ----------
        pos_texts:
            Positive (target-direction) prompts.
        neg_texts:
            Negative (opposite-direction) prompts.
        max_length:
            Tokenization cap.

        Returns
        -------
        torch.Tensor
            Steering vector of shape ``(hidden_dim,)``.  Also stored as
            ``self.steering_vector``.
        """
        if self.tokenizer is None:
            raise RuntimeError("CAAInjector.calibrate: tokenizer must be set first.")
        if not pos_texts or not neg_texts:
            raise ValueError("calibrate: pos_texts and neg_texts must be non-empty.")

        layer_idx = self._resolve_layer()

        def _collect_hidden(texts: list[str]) -> torch.Tensor:
            """Return mean hidden state at layer_idx over all texts, shape (D,)."""
            accum: Optional[torch.Tensor] = None
            count = 0
            for text in texts:
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = enc["input_ids"].to(self._device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._device)

                # Capture hidden state at target layer via a one-shot hook.
                captured: list[torch.Tensor] = []

                def _cap_hook(module: Any, inp: Any, output: Any) -> None:
                    h = output[0] if isinstance(output, tuple) else output
                    captured.append(h.detach().float().mean(dim=(0, 1)))  # (D,)

                layers = self._get_decoder_layers()
                handle = layers[layer_idx].register_forward_hook(_cap_hook)
                try:
                    was_training = bool(getattr(self._model, "training", False))
                    self._model.eval()
                    with torch.no_grad():
                        self._model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                finally:
                    handle.remove()
                    if was_training:
                        self._model.train()

                if captured:
                    h_mean = captured[0]  # (D,)
                    accum = h_mean if accum is None else accum + h_mean
                    count += 1

            if accum is None or count == 0:
                raise RuntimeError("calibrate: no hidden states captured.")
            return accum / count  # mean over texts

        pos_mean = _collect_hidden(pos_texts)
        neg_mean = _collect_hidden(neg_texts)
        s = pos_mean - neg_mean  # (D,)

        self.steering_vector = s.to(self._device)
        return self.steering_vector

    # ------------------------------------------------------------------
    # Context-manager (hook installation / removal)
    # ------------------------------------------------------------------

    def __enter__(self) -> "CAAInjector":
        if self.steering_vector is None:
            raise RuntimeError(
                "CAAInjector: call calibrate() or set steering_vector before entering context."
            )

        alpha = float(self.config.alpha)
        layer_idx = self._resolve_layer()
        s = self.steering_vector  # (D,)

        # Reset gate state at session boundary.
        self._prev_hidden = None

        use_gate = self.config.use_lopi_gate
        gate_k = float(self.config.gate_k)
        gate_theta = float(self.config.gate_theta)

        def _hook(module: Any, inp: Any, output: Any) -> Any:
            if alpha == 0.0:
                # Bit-equal identity — do not touch the output.
                return output

            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output  # (B, T, D)

            if use_gate:
                # γ_t = sigmoid(k * (‖h_t − h_{t-1}‖₂ − θ))
                # Shape: (B, T, 1) for broadcasting.
                prev = self._prev_hidden
                if prev is None or prev.shape != hidden.shape:
                    gamma = torch.ones(
                        *hidden.shape[:-1], 1,
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                else:
                    delta_h = torch.linalg.vector_norm(
                        hidden.float() - prev.float(),
                        ord=2, dim=-1, keepdim=True,
                    )  # (B, T, 1)
                    arg = (delta_h - gate_theta) * gate_k
                    gamma = torch.sigmoid(arg).to(hidden.dtype)
                self._prev_hidden = hidden.detach()
            else:
                gamma = 1.0  # scalar — no allocation, no gate

            # s: (D,) → broadcast to (1, 1, D)
            s_bc = s.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(0).unsqueeze(0)
            new_hidden = hidden + alpha * gamma * s_bc

            if is_tuple:
                return (new_hidden,) + output[1:]
            return new_hidden

        layers = self._get_decoder_layers()
        self._hook_handle = layers[layer_idx].register_forward_hook(_hook)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._prev_hidden = None


__all__ = [
    "CAAConfig",
    "CAAInjector",
]
