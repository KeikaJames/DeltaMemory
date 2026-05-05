"""Phase X.1 — DiagnosticRecorder for 5 internal Mneme signals.

Usage::

    with DiagnosticRecorder(model, patcher, lopi_state=state, enabled=True) as rec:
        out = model(input_ids)
    df = rec.to_pandas()
    rec.dump_parquet("path/to/out.parquet")

The recorder is a thin context manager that:

1. Registers a model-level ``forward_pre_hook`` to count steps (one step =
   one full model forward).
2. Registers a per-block ``forward_hook`` on each decoder layer to record
   residual-stream L2 norms per token.
3. Sets the module-level ``_RECORDER`` global so hook points inside
   ``attn_native_bank.py`` and ``lopi.py`` can deposit signals with a single
   ``if _RECORDER is not None`` guard — zero overhead when recorder is off.

Signals recorded (all in long-format rows)
------------------------------------------
* ``bank_col_sum``        — col-sum of weights[..., T_orig:] per bank slot,
                            captured **before** the mHC shield clip.
* ``attn_entropy_native`` — Shannon entropy of the native-context attention
                            slice (averaged over B × H × T).
* ``attn_entropy_bank``   — Shannon entropy of the bank attention slice
                            (averaged over B × H × T).
* ``lopi_gamma_t``        — LOPI derivative gate value per token (requires
                            LOPI enabled AND ``lopi_state`` provided).
* ``lopi_w_ell``          — LOPI layer Gaussian weight (requires LOPI enabled).
* ``m_perp_energy_ratio`` — ‖M_⊥‖² / ‖M_V‖² at the orthogonal-projection
                            site inside lopi.py (requires LOPI enabled).
* ``residual_norm``       — L2 norm of the residual stream per token, per layer.

DataFrame columns
-----------------
``step`` (int32), ``layer`` (int32), ``token`` (int32; -1 = no token dim),
``signal_name`` (object), ``value`` (float32).

Red-line guarantee
------------------
When ``enabled=False`` (or when no recorder is active) every hook is either
absent (registered hooks) or behind a falsy guard (``if _RECORDER is not
None``).  The α=0 bit-equal property is preserved unconditionally.
"""
from __future__ import annotations

from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Module-level singleton accessed by hook points in attn_native_bank.py and
# lopi.py.  Falsy when no recorder is active → near-zero overhead guard.
# ---------------------------------------------------------------------------

_RECORDER: Optional["DiagnosticRecorder"] = None


# ---------------------------------------------------------------------------
# DiagnosticRecorder
# ---------------------------------------------------------------------------

class DiagnosticRecorder:
    """Context-manager that records 5 internal Mneme signals per step.

    Parameters
    ----------
    model:
        The HuggingFace causal-LM model to monitor (e.g. Llama, Gemma, Qwen).
    patcher:
        The :class:`~deltamemory.memory.attn_native_bank.AttnNativePatcher`
        attached to *model*.  Used only to locate decoder layer modules for
        residual-norm hooks.
    lopi_state:
        :class:`~deltamemory.memory.lopi.LOPIState` associated with the bank.
        When *None*, LOPI-specific signals (``lopi_gamma_t``, ``lopi_w_ell``)
        are silently skipped.
    enabled:
        Set to *False* to make the entire recorder a no-op (context manager
        still works, ``to_pandas`` returns an empty DataFrame).
    """

    def __init__(
        self,
        model: Any,
        patcher: Any,
        lopi_state: Any = None,
        enabled: bool = True,
    ) -> None:
        self._model = model
        self._patcher = patcher
        self._lopi_state = lopi_state
        self._enabled = enabled
        self._records: list[dict] = []
        self._current_step: int = -1
        self._hook_handles: list = []
        self._previous_recorder: Optional["DiagnosticRecorder"] = None
        self._entered: bool = False

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "DiagnosticRecorder":
        if not self._enabled:
            return self
        global _RECORDER
        self._previous_recorder = _RECORDER
        _RECORDER = self

        try:
            # Model-level pre-hook: increment step counter once per forward.
            def _pre_hook(module: Any, args: Any) -> None:
                self._current_step += 1

            self._hook_handles.append(
                self._model.register_forward_pre_hook(_pre_hook)
            )

            # Per-decoder-block hook: record residual-stream L2 norms.
            for i, block in enumerate(self._find_decoder_layers()):
                self._hook_handles.append(
                    block.register_forward_hook(self._make_residual_hook(i))
                )
        except Exception:
            for h in self._hook_handles:
                h.remove()
            self._hook_handles.clear()
            _RECORDER = self._previous_recorder
            self._previous_recorder = None
            raise

        self._entered = True
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._enabled or not self._entered:
            return
        global _RECORDER
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        _RECORDER = self._previous_recorder
        self._previous_recorder = None
        self._entered = False

    # ------------------------------------------------------------------
    # Decoder-layer discovery (mirrors AttnNativePatcher.__init__ paths)
    # ------------------------------------------------------------------

    def _find_decoder_layers(self) -> list:
        """Return the list of decoder block modules (not attn submodules)."""
        model = self._model
        for path in (
            "model.model.language_model.layers",
            "model.model.layers",
            "model.language_model.model.layers",
            "model.language_model.layers",
            "language_model.layers",
            "model.layers",
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
            "DiagnosticRecorder: could not locate decoder layers on the model. "
            "Ensure the model has a standard HuggingFace decoder-layer structure."
        )

    def _make_residual_hook(self, layer_idx: int):
        """Build a ``forward_hook`` that records residual L2 norms per token."""
        def _hook(module: Any, inp: Any, output: Any) -> None:
            step = self._current_step
            if step < 0:
                return
            # Output may be a tuple (hidden_states, ...) or a plain tensor.
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (B, T, D) → per-token L2 norm.
            norms = torch.linalg.vector_norm(
                hidden.detach().float(), ord=2, dim=-1
            )  # (B, T)
            norms_mean = norms.mean(dim=0)  # (T,) averaged over batch
            for t in range(norms_mean.size(0)):
                self._records.append({
                    "step": step,
                    "layer": layer_idx,
                    "token": t,
                    "signal_name": "residual_norm",
                    "value": float(norms_mean[t].item()),
                })
        return _hook

    # ------------------------------------------------------------------
    # Hook-point methods called from attn_native_bank.py and lopi.py
    # ------------------------------------------------------------------

    def record_bank_attn(
        self,
        layer_idx: int,
        weights: torch.Tensor,
        T_orig: int,
    ) -> None:
        """Record bank-attention signals (col-sum, entropies).

        Called from the patched forward in ``attn_native_bank.py`` after the
        merged-softmax but **before** the mHC shield clip.

        Parameters
        ----------
        weights:
            Post-softmax weight tensor, shape ``(B, H, T, T_orig+N)``.
        T_orig:
            Number of native-sequence columns (first ``T_orig`` columns).
        """
        step = self._current_step
        if step < 0:
            return
        w = weights.detach().float()  # (B, H, T, T_orig+N)

        # 1. bank_col_sum — sum over (B, H, T) per bank slot.
        w_bank = w[..., T_orig:]  # (B, H, T, N)
        col_sum = w_bank.sum(dim=(0, 1, 2))  # (N,)
        for j in range(col_sum.size(0)):
            self._records.append({
                "step": step,
                "layer": layer_idx,
                "token": j,          # bank-slot index stored in token column
                "signal_name": "bank_col_sum",
                "value": float(col_sum[j].item()),
            })

        # 2. Shannon entropy of native and bank attention slices.
        _eps = 1e-10
        w_nat = w[..., :T_orig]  # (B, H, T, T_orig)
        # Re-normalise within each slice for valid probability distributions.
        w_nat_p = w_nat / (w_nat.sum(dim=-1, keepdim=True) + _eps)
        ent_nat = -(w_nat_p * (w_nat_p + _eps).log()).sum(dim=-1)  # (B, H, T)

        w_bnk_p = w_bank / (w_bank.sum(dim=-1, keepdim=True) + _eps)
        ent_bnk = -(w_bnk_p * (w_bnk_p + _eps).log()).sum(dim=-1)  # (B, H, T)

        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "attn_entropy_native",
            "value": float(ent_nat.mean().item()),
        })
        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "attn_entropy_bank",
            "value": float(ent_bnk.mean().item()),
        })

    def record_lopi_gamma_w(
        self,
        layer_idx: int,
        gamma_t: torch.Tensor,
        w_ell: torch.Tensor,
    ) -> None:
        """Record ``lopi_gamma_t`` (per token) and ``lopi_w_ell`` (scalar).

        Called from :func:`deltamemory.memory.lopi.apply_lopi` after
        ``gamma_t`` and ``w_ell`` are computed.  Silently skipped when
        ``self._lopi_state is None``.
        """
        if self._lopi_state is None:
            return
        step = self._current_step
        if step < 0:
            return

        # gamma_t: (B, H, T, 1) or 0-dim scalar tensor.
        g = gamma_t.detach().float()
        if g.dim() >= 3:
            # Collapse B and H dimensions, squeeze trailing 1 → (T,).
            g = g.mean(dim=(0, 1)).squeeze(-1)
            for t in range(g.size(0)):
                self._records.append({
                    "step": step,
                    "layer": layer_idx,
                    "token": t,
                    "signal_name": "lopi_gamma_t",
                    "value": float(g[t].item()),
                })
        else:
            self._records.append({
                "step": step,
                "layer": layer_idx,
                "token": 0,
                "signal_name": "lopi_gamma_t",
                "value": float(g.item()),
            })

        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "lopi_w_ell",
            "value": float(w_ell.detach().float().item()),
        })

    def record_m_perp_ratio(
        self,
        layer_idx: int,
        m_perp: torch.Tensor,
        m_v: torch.Tensor,
    ) -> None:
        """Record ‖M_⊥‖² / ‖M_V‖² at the orthogonal-projection site.

        Called from :func:`deltamemory.memory.lopi.apply_lopi` immediately
        after the orthogonal-novelty projection step (step 1).  When
        ``cfg.orthogonal=False``, ``m_perp is m_v`` and the ratio is 1.0.
        """
        step = self._current_step
        if step < 0:
            return
        m_perp_sq = float((m_perp.detach().float() ** 2).sum().item())
        m_v_sq = float((m_v.detach().float() ** 2).sum().item())
        ratio = m_perp_sq / (m_v_sq + 1e-10)
        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "m_perp_energy_ratio",
            "value": float(ratio),
        })

    def record_caa_steer(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        hidden: torch.Tensor,
        alpha: float,
        gamma: Optional[torch.Tensor] = None,
    ) -> None:
        """Record CAA steering diagnostics at the injection site.

        Emits three signals at token=-1 for the current (step, layer):

        - ``caa_steer_norm``         = ``alpha * ||s||``
        - ``caa_gate_mean``          = mean of ``gamma`` (1.0 when ungated)
        - ``caa_hidden_drift_ratio`` = ``||alpha * gamma * s||_F / ||hidden||_F``

        Mirrors :meth:`record_scar_proj` and ``record_lopi_gamma_w`` so
        downstream W.12 ablation can compare LOPI / SCAR / CAA telemetry on
        a uniform contract.  Silently skipped when no recorder is active.
        """
        step = self._current_step
        if step < 0:
            return
        s = steering_vector.detach().float()
        h = hidden.detach().float()
        s_norm = float(torch.linalg.vector_norm(s).item())
        h_norm = float(torch.linalg.vector_norm(h).item())

        if gamma is None:
            gamma_mean = 1.0
            perturb_norm = abs(alpha) * s_norm
        else:
            g = gamma.detach().float()
            gamma_mean = float(g.mean().item())
            # ||alpha * gamma * s||_F over (B, T, D) broadcast.
            perturb = (alpha * g) * s.view(*([1] * (g.dim() - 1)), -1)
            perturb_norm = float(torch.linalg.vector_norm(perturb).item())

        drift_ratio = perturb_norm / (h_norm + 1e-10)

        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "caa_steer_norm",
            "value": float(abs(alpha) * s_norm),
        })
        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "caa_gate_mean",
            "value": float(gamma_mean),
        })
        self._records.append({
            "step": step,
            "layer": layer_idx,
            "token": -1,
            "signal_name": "caa_hidden_drift_ratio",
            "value": float(drift_ratio),
        })

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def to_pandas(self):
        """Assemble collected signals into a long-format :class:`pandas.DataFrame`.

        Columns: ``step`` (int32), ``layer`` (int32), ``token`` (int32),
        ``signal_name`` (object), ``value`` (float32).
        """
        import pandas as pd

        if not self._records:
            return (
                pd.DataFrame(
                    columns=["step", "layer", "token", "signal_name", "value"]
                )
                .astype({
                    "step": "int32",
                    "layer": "int32",
                    "token": "int32",
                    "value": "float32",
                })
            )

        df = pd.DataFrame(self._records)
        df["step"] = df["step"].astype("int32")
        df["layer"] = df["layer"].astype("int32")
        df["token"] = df["token"].astype("int32")
        df["value"] = df["value"].astype("float32")
        return df

    def dump_parquet(self, path: str) -> None:
        """Write collected signals to Parquet (pyarrow) or JSON-lines fallback.

        If ``pyarrow`` is not installed the DataFrame is written as a
        newline-delimited JSON file to *path*.
        """
        df = self.to_pandas()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, path)
        except ImportError:
            df.to_json(path, orient="records", lines=True)


__all__ = ["DiagnosticRecorder", "_RECORDER"]
