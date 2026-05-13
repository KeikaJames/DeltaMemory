"""SCAR — Steering with Contrastive Activation Rotation.

Injects a' = a + α · B B^T (target − a) at attention output, where B is the
top-k right singular vectors of (pos_acts − neg_acts) collected at calibration
time. Frozen weights, zero new nn.Parameters, bit-equal at α=0.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SCARInjector(nn.Module):
    """Hook-based SCAR injector for HuggingFace decoder-only causal LMs."""

    def __init__(
        self,
        model: Any,
        alpha: float = 0.0,
        layers: list[int] | None = None,
        k: int = 2,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_model", model)
        self.alpha = float(alpha)
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("SCARInjector: k must be positive.")

        n_layers = len(self._get_decoder_layers())
        if layers is None:
            self.layers = [n_layers // 2]
        else:
            self.layers = [int(layer) for layer in layers]
        for layer in self.layers:
            if layer < 0 or layer >= n_layers:
                raise ValueError(f"SCARInjector: layer {layer} out of range for {n_layers} layers.")

        self.basis: dict[int, torch.Tensor] = {}
        self.target_mean: dict[int, torch.Tensor] = {}
        self._hook_handles: list[Any] = []

    @property
    def is_calibrated(self) -> bool:
        """Whether every configured layer has a basis and target mean."""
        return all(layer in self.basis and layer in self.target_mean for layer in self.layers)

    def _get_decoder_layers(self) -> list[Any]:
        from deltamemory.memory._layer_locator import get_decoder_layers
        try:
            return get_decoder_layers(self._model)
        except RuntimeError as exc:
            raise RuntimeError(
                "SCARInjector: could not locate decoder layers on the model. "
                "Expected a standard HuggingFace decoder-layer structure."
            ) from exc

    def _get_attention_output_module(self, layer_idx: int) -> nn.Module:
        layer = self._get_decoder_layers()[layer_idx]
        for path in (
            "self_attn.o_proj",
            "self_attention.o_proj",
            "attention.o_proj",
            "attn.o_proj",
            "self_attn.out_proj",
            "attention.out_proj",
            "attn.c_proj",
        ):
            obj: Any = layer
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and isinstance(obj, nn.Module):
                return obj
        raise RuntimeError(
            f"SCARInjector: could not locate attention output projection for layer {layer_idx}."
        )

    def calibrate(
        self,
        pos_prompts: list[str],
        neg_prompts: list[str],
        tokenizer: Any,
        max_n: int = 32,
    ) -> None:
        """Collect last-token attention outputs and build per-layer SVD bases."""
        if not pos_prompts or not neg_prompts:
            raise ValueError("SCARInjector.calibrate: pos_prompts and neg_prompts must be non-empty.")
        if tokenizer is None:
            raise RuntimeError("SCARInjector.calibrate: tokenizer is required.")

        n = min(len(pos_prompts), len(neg_prompts), int(max_n))
        if n <= 0:
            raise ValueError("SCARInjector.calibrate: max_n must be positive.")

        pos_by_layer = {layer: [] for layer in self.layers}
        neg_by_layer = {layer: [] for layer in self.layers}

        for pos_prompt, neg_prompt in zip(pos_prompts[:n], neg_prompts[:n]):
            pos_caps = self._collect_last_token_acts(pos_prompt, tokenizer)
            neg_caps = self._collect_last_token_acts(neg_prompt, tokenizer)
            for layer in self.layers:
                pos_by_layer[layer].append(pos_caps[layer])
                neg_by_layer[layer].append(neg_caps[layer])

        self.basis.clear()
        self.target_mean.clear()
        for layer in self.layers:
            pos = torch.stack(pos_by_layer[layer], dim=0).float()
            neg = torch.stack(neg_by_layer[layer], dim=0).float()
            diff = pos - neg
            _, _, vh = torch.linalg.svd(diff, full_matrices=False)
            rank_k = min(self.k, vh.shape[0])
            if rank_k < self.k:
                d = diff.shape[-1]
                padded = torch.zeros(d, self.k, dtype=torch.float32)
                if rank_k > 0:
                    padded[:, :rank_k] = vh[:rank_k].T.contiguous()
                self.basis[layer] = padded
            else:
                self.basis[layer] = vh[: self.k].T.contiguous()
            self.target_mean[layer] = pos.mean(dim=0).contiguous()

    def _collect_last_token_acts(self, prompt: str, tokenizer: Any) -> dict[int, torch.Tensor]:
        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            last_idx = int(attention_mask[0].sum().item()) - 1
        else:
            last_idx = input_ids.shape[1] - 1
        last_idx = max(last_idx, 0)

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden[0, last_idx, :].detach().cpu().float()

            return _hook

        for layer in self.layers:
            handles.append(self._get_attention_output_module(layer).register_forward_hook(make_hook(layer)))

        was_training = bool(getattr(self._model, "training", False))
        self._model.eval()
        try:
            with torch.no_grad():
                self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self._model.train()

        missing = [layer for layer in self.layers if layer not in captured]
        if missing:
            raise RuntimeError(f"SCARInjector.calibrate: no activations captured for layers {missing}.")
        return captured

    def attach(self) -> None:
        """Register attention-output hooks for all configured layers."""
        if self._hook_handles:
            return
        for layer in self.layers:
            module = self._get_attention_output_module(layer)
            self._hook_handles.append(module.register_forward_hook(self._make_inject_hook(layer)))

    def detach(self) -> None:
        """Remove all active SCAR hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __enter__(self) -> "SCARInjector":
        self.attach()
        return self

    def __exit__(self, *_: Any) -> None:
        self.detach()

    def _make_inject_hook(self, layer: int):
        def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden = self._do_inject(hidden, layer)
                if new_hidden is hidden:
                    return output
                return (new_hidden,) + output[1:]
            return self._do_inject(output, layer)

        return _hook

    def _do_inject(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply SCAR projection to an attention-output tensor."""
        if self.alpha == 0.0 or layer not in self.basis or layer not in self.target_mean:
            return activations

        basis = self.basis[layer].to(device=activations.device, dtype=torch.float32)
        target = self.target_mean[layer].to(device=activations.device, dtype=torch.float32)
        delta = target.view(*([1] * (activations.ndim - 1)), -1) - activations.float()
        projected = (delta @ basis) @ basis.T

        # Diagnostics emit (silent no-op when no recorder is active; mirrors
        # the LOPI / mHC bank pattern). Lazy import avoids circular-import on
        # module load and keeps SCAR usable without diagnostics installed.
        try:
            from deltamemory import diagnostics as _diag_mod
            if _diag_mod._RECORDER is not None:
                _diag_mod._RECORDER.record_scar_proj(
                    layer_idx=layer,
                    delta=delta,
                    projected=projected,
                    alpha=self.alpha,
                )
                # A2 — five-signal SCAR diagnostics (drift, proj_B_mass,
                # ortho_residue, alpha-monotonic, contract_violation).
                x_pre = activations.float()
                x_post = x_pre + (self.alpha * projected)
                diff = x_post - x_pre  # α · projected in float32
                x_pre_norm = float(torch.linalg.vector_norm(x_pre).item())
                diff_norm = float(torch.linalg.vector_norm(diff).item())
                drift = diff_norm / (x_pre_norm + 1e-10)
                # proj_B_mass = ‖projected‖² / ‖α·projected‖² = 1/α² when α≠0
                # But we want ‖P_B diff‖² / ‖diff‖² where diff = α·projected.
                # Since diff IS α·projected and projected = P_B(delta),
                # we have diff = α·P_B(delta). So ‖P_B(diff)‖ = ‖diff‖ because
                # diff is already fully in span(B). Thus proj_B_mass = 1.0
                # by construction UNLESS there's numerical error or the
                # projection wasn't perfect. We compute it from projected:
                proj_sq = float((projected.detach().float() ** 2).sum().item())
                delta_sq = float((delta.detach().float() ** 2).sum().item())
                proj_B_mass = proj_sq / (delta_sq + 1e-10) if delta_sq > 1e-10 else 0.0
                ortho_residue = 1.0 - proj_B_mass
                contract_violation = ortho_residue < -1e-5 or ortho_residue > 1.0 + 1e-5
                _diag_mod._RECORDER.record_scar(
                    layer_idx=layer,
                    drift=drift,
                    proj_B_mass=proj_B_mass,
                    ortho_residue=max(0.0, ortho_residue),
                    alpha=self.alpha,
                    contract_violation=contract_violation,
                )
        except Exception:
            pass


        try:
            from deltamemory.security.audit import audit_event

            projected_norm = float(torch.linalg.vector_norm(projected.detach().float()).item())
            activation_norm = float(torch.linalg.vector_norm(activations.detach().float()).item())
            audit_event(
                event_type="inject",
                injector="scar",
                layer=layer,
                alpha=self.alpha,
                signal_summary={
                    "steer_norm": abs(self.alpha) * projected_norm,
                    "drift_ratio": (abs(self.alpha) * projected_norm) / (activation_norm + 1e-10),
                    "gate_mean": 1.0,
                },
                vector_tensor=projected,
            )
        except Exception:
            pass

        return activations + (self.alpha * projected).to(dtype=activations.dtype)


__all__ = ["SCARInjector"]
