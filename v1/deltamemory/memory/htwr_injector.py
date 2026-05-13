"""Hard-Top1 Whitened Retrieval (HTWR) — Exp12.

Builds on the RSM idea (capture last-token residual per layer, replay into the
read prompt's residual stream) but abandons soft cosine weighting.  Exp12 always
injects exactly one memory per read prompt:

    i*       = retriever(query_residuals, bank)
    h_ℓ     ← h_ℓ + η · m_{i*, ℓ}            for ℓ in inject_layers

Retrievers are pluggable so we can climb the T0–T4 ladder without changing the
injection path:

    OracleRetriever         — pick by fact_id match (ceiling test)
    RandomRetriever         — pick uniformly at random
    RawCosineRetriever      — max-over-layers cosine (T1)
    WhitenedCosineRetriever — ZCA/diag/PCA + cosine (T2)
    CompositionalRetriever  — multi-key score aggregation (T3)
    ProjectorRetriever      — learned linear/MLP projector (T4)

Variants like ``oracle_shuffled`` and ``oracle_sign_flip`` are implemented as
*bank transforms* applied before retrieval, so the injector itself stays small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import torch
import torch.nn.functional as F

from deltamemory.memory._layer_locator import get_decoder_layers

HTWRHookPoint = Literal["block_output", "pre_block_input", "mlp_mid"]


@dataclass
class HTWRConfig:
    """Hyperparameters for capture/inject. Retriever selection is separate."""

    eta: float = 0.05
    hook_point: HTWRHookPoint = "pre_block_input"
    inject_only_last_token: bool = True
    # If None, inject at every layer; else use the explicit (ordered) indices.
    inject_layers: tuple[int, ...] | None = None
    # Multiplier applied to the injected memory before adding to h.
    # Useful for "sign_flip" (-1) as a control without rebuilding the bank.
    inject_sign: float = 1.0


@dataclass
class HTWRMemoryBank:
    """Per-fact, per-layer residual memories with optional auxiliary keys."""

    memories: torch.Tensor             # (N, L, D_value) — primary residual bank
    fact_ids: list[str]
    keys: torch.Tensor | None = None   # (N, L, D_key) — optional separate keys
    subjects: list[str] | None = None
    relations: list[str] | None = None
    targets: list[str] | None = None
    hook_point: str = "pre_block_input"
    key_type: str = "last_token"

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("memories must have shape (N, L, D).")
        if self.memories.shape[0] != len(self.fact_ids):
            raise ValueError("fact_ids length mismatch with memories.shape[0].")
        self.memories = self.memories.detach().cpu().float().contiguous()
        self.fact_ids = [str(fid) for fid in self.fact_ids]
        if self.keys is not None:
            if self.keys.ndim != 3 or self.keys.shape[0] != self.memories.shape[0]:
                raise ValueError("keys must have shape (N, L, D_key).")
            self.keys = self.keys.detach().cpu().float().contiguous()

    @property
    def n_memories(self) -> int:
        return int(self.memories.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.memories.shape[1])

    @property
    def value_dim(self) -> int:
        return int(self.memories.shape[2])

    @property
    def key_dim(self) -> int:
        return int(self.keys.shape[2]) if self.keys is not None else self.value_dim

    def shuffled_layers(self, *, seed: int = 0xA11CE) -> "HTWRMemoryBank":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(self.n_layers, generator=gen)
        return HTWRMemoryBank(
            memories=self.memories[:, perm, :].contiguous(),
            fact_ids=list(self.fact_ids),
            keys=self.keys[:, perm, :].contiguous() if self.keys is not None else None,
            subjects=list(self.subjects) if self.subjects else None,
            relations=list(self.relations) if self.relations else None,
            targets=list(self.targets) if self.targets else None,
            hook_point=self.hook_point,
            key_type=self.key_type,
        )

    def index(self, idx: int) -> torch.Tensor:
        return self.memories[idx]


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    top_index: int
    top_score: float
    correct_score: float
    top_minus_mean: float
    mean_score: float
    score_std: float
    retrieval_accuracy: bool
    all_scores: torch.Tensor  # (N,) on CPU


class HTWRRetriever(Protocol):
    """Picks one bank index from a per-layer query residual stack."""

    name: str

    def retrieve(
        self,
        query_residuals: torch.Tensor,  # (L, D_key) CPU fp32
        bank: HTWRMemoryBank,
        *,
        correct_fact_id: str | None = None,
    ) -> RetrievalResult: ...


def _empty_result(correct_known: bool) -> RetrievalResult:
    return RetrievalResult(
        top_index=-1,
        top_score=float("nan"),
        correct_score=float("nan"),
        top_minus_mean=float("nan"),
        mean_score=float("nan"),
        score_std=float("nan"),
        retrieval_accuracy=False,
        all_scores=torch.empty(0),
    )


def _maxlayer_cosine(query: torch.Tensor, bank_keys: torch.Tensor) -> torch.Tensor:
    """query (L, D), bank_keys (N, L, D) → (N,) max-over-layers cosine."""
    if query.shape[0] != bank_keys.shape[1]:
        raise ValueError(
            f"query layers ({query.shape[0]}) != bank layers ({bank_keys.shape[1]})"
        )
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(bank_keys.float(), dim=-1)
    # per-layer dot: (N, L)
    per_layer = torch.einsum("ld,nld->nl", q, k)
    return per_layer.max(dim=1).values


def _result_from_scores(
    scores: torch.Tensor,
    bank: HTWRMemoryBank,
    correct_fact_id: str | None,
) -> RetrievalResult:
    top_idx = int(scores.argmax().item())
    top_score = float(scores[top_idx].item())
    mean_score = float(scores.mean().item())
    score_std = float(scores.std(unbiased=False).item())
    if correct_fact_id is None or correct_fact_id not in bank.fact_ids:
        correct_score = float("nan")
        accuracy = False
    else:
        correct_idx = bank.fact_ids.index(correct_fact_id)
        correct_score = float(scores[correct_idx].item())
        accuracy = top_idx == correct_idx
    return RetrievalResult(
        top_index=top_idx,
        top_score=top_score,
        correct_score=correct_score,
        top_minus_mean=top_score - mean_score,
        mean_score=mean_score,
        score_std=score_std,
        retrieval_accuracy=accuracy,
        all_scores=scores.detach().cpu(),
    )


@dataclass
class OracleRetriever:
    """Picks the bank entry whose fact_id matches `correct_fact_id`."""

    name: str = "oracle"

    def retrieve(
        self,
        query_residuals: torch.Tensor,
        bank: HTWRMemoryBank,
        *,
        correct_fact_id: str | None = None,
    ) -> RetrievalResult:
        if correct_fact_id is None or correct_fact_id not in bank.fact_ids:
            return _empty_result(correct_known=False)
        n = bank.n_memories
        idx = bank.fact_ids.index(correct_fact_id)
        scores = torch.zeros(n)
        scores[idx] = 1.0
        return _result_from_scores(scores, bank, correct_fact_id)


@dataclass
class RandomRetriever:
    seed: int = 0
    name: str = "random"

    def retrieve(
        self,
        query_residuals: torch.Tensor,
        bank: HTWRMemoryBank,
        *,
        correct_fact_id: str | None = None,
    ) -> RetrievalResult:
        gen = torch.Generator().manual_seed(self.seed)
        scores = torch.rand(bank.n_memories, generator=gen)
        return _result_from_scores(scores, bank, correct_fact_id)


@dataclass
class RawCosineRetriever:
    """Max-over-layers cosine on raw (un-whitened) bank keys/values."""

    name: str = "raw_cosine"

    def retrieve(
        self,
        query_residuals: torch.Tensor,
        bank: HTWRMemoryBank,
        *,
        correct_fact_id: str | None = None,
    ) -> RetrievalResult:
        keys = bank.keys if bank.keys is not None else bank.memories
        scores = _maxlayer_cosine(query_residuals, keys)
        return _result_from_scores(scores, bank, correct_fact_id)


@dataclass
class WhitenedCosineRetriever:
    """Score in whitened key space; inject path uses original memories.

    Whitener is built externally (see :mod:`htwr_whitening`) and passed in.
    Expected shapes:
        mu_per_layer : (L, D_key)
        W_per_layer  : (L, D_key, D_key) or (L, K, D_key) for PCA-K.
    The query and bank keys are transformed as:
        x̃ = W (x − μ)
    """

    mu_per_layer: torch.Tensor
    W_per_layer: torch.Tensor
    name: str = "whitened_cosine"

    def retrieve(
        self,
        query_residuals: torch.Tensor,
        bank: HTWRMemoryBank,
        *,
        correct_fact_id: str | None = None,
    ) -> RetrievalResult:
        keys = bank.keys if bank.keys is not None else bank.memories
        q = query_residuals.float()
        # (L, D_key) − (L, D_key) → (L, D_key); then per-layer matmul: (L, K)
        q_centered = q - self.mu_per_layer
        # einsum over per-layer matrices
        q_proj = torch.einsum("lkd,ld->lk", self.W_per_layer, q_centered)
        k_centered = keys - self.mu_per_layer.unsqueeze(0)
        k_proj = torch.einsum("lkd,nld->nlk", self.W_per_layer, k_centered)
        scores = _maxlayer_cosine(q_proj, k_proj)
        return _result_from_scores(scores, bank, correct_fact_id)


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------


def _as_hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def _last_index(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> int:
    if input_ids.shape[0] != 1:
        raise ValueError("HTWR currently supports batch_size=1.")
    if attention_mask is not None:
        return max(0, int(attention_mask[0].sum().item()) - 1)
    return max(0, int(input_ids.shape[1]) - 1)


def _resolve_target_module(layer: Any, hook_point: HTWRHookPoint) -> Any:
    if hook_point in ("block_output", "pre_block_input"):
        return layer
    if hook_point == "mlp_mid":
        mlp = getattr(layer, "mlp", None)
        down = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down is None:
            raise ValueError(
                "hook_point='mlp_mid' requires layer.mlp.down_proj on every "
                "decoder layer (SwiGLU-style MLPs)."
            )
        return down
    raise ValueError(f"Unknown hook_point: {hook_point!r}")


def _is_pre_hook(hook_point: HTWRHookPoint) -> bool:
    return hook_point in ("pre_block_input", "mlp_mid")


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class HTWRInjector:
    """Hard-top-1 residual stream memory injector."""

    def __init__(self, model: Any, config: HTWRConfig | None = None) -> None:
        self.model = model
        self.config = config or HTWRConfig()
        self.layers = get_decoder_layers(model)
        # Validate hook_point against all layers up front to fail loudly.
        for layer in self.layers:
            _resolve_target_module(layer, self.config.hook_point)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _target_modules(self) -> list[Any]:
        return [_resolve_target_module(l, self.config.hook_point) for l in self.layers]

    # ----- capture ---------------------------------------------------------

    @torch.no_grad()
    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        token_index: int | None = None,
    ) -> torch.Tensor:
        """Capture per-layer activation at ``token_index`` (last token by default).

        Returns ``(L, D)`` CPU fp32 tensor.
        """
        target_idx = token_index if token_index is not None else _last_index(
            input_ids, attention_mask
        )
        captured: dict[int, torch.Tensor] = {}
        handles = []
        pre_hook = _is_pre_hook(self.config.hook_point)

        def make_post(layer_idx: int):
            def _hook(module, inputs, output):
                h = _as_hidden(output)
                captured[layer_idx] = h[0, target_idx, :].detach().cpu().float()
            return _hook

        def make_pre(layer_idx: int):
            def _hook(module, inputs):
                h = inputs[0]
                captured[layer_idx] = h[0, target_idx, :].detach().cpu().float()
                return None
            return _hook

        targets = self._target_modules()
        for idx, mod in enumerate(targets):
            if pre_hook:
                handles.append(mod.register_forward_pre_hook(make_pre(idx)))
            else:
                handles.append(mod.register_forward_hook(make_post(idx)))

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()

        if len(captured) != self.num_layers:
            raise RuntimeError(
                f"capture: got {len(captured)} layers, expected {self.num_layers}."
            )
        return torch.stack([captured[i] for i in range(self.num_layers)], dim=0)

    # ----- query (one forward, no mutation) --------------------------------

    @torch.no_grad()
    def query_residuals(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same as capture(), exposed as a semantic alias for retrieval scoring."""
        return self.capture(input_ids, attention_mask)

    # ----- inject ----------------------------------------------------------

    def forward_with_memory(
        self,
        memory: torch.Tensor,  # (L, D)
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run a forward pass with ``memory`` added to the chosen hook point.

        ``memory`` is the single bank entry chosen by retrieval (or by oracle).
        Applies ``config.inject_sign * config.eta`` and respects ``inject_layers``
        if set.
        """
        if memory.ndim != 2 or memory.shape[0] != self.num_layers:
            raise ValueError(
                f"memory must be (L, D); got {tuple(memory.shape)} for L={self.num_layers}."
            )
        eta = float(self.config.eta) * float(self.config.inject_sign)
        last_idx = _last_index(input_ids, attention_mask)
        only_last = bool(self.config.inject_only_last_token)
        inject_set = (
            set(range(self.num_layers))
            if self.config.inject_layers is None
            else set(int(i) for i in self.config.inject_layers)
        )
        pre_hook = _is_pre_hook(self.config.hook_point)
        handles = []
        targets = self._target_modules()

        def _apply(hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if eta == 0.0 or layer_idx not in inject_set:
                return hidden
            delta = memory[layer_idx].to(device=hidden.device, dtype=hidden.dtype) * eta
            if only_last:
                hidden = hidden.clone()
                hidden[0, last_idx, :] = hidden[0, last_idx, :] + delta
            else:
                hidden = hidden + delta
            return hidden

        def make_post(layer_idx: int):
            def _hook(module, inputs, output):
                h = _as_hidden(output)
                h_new = _apply(h, layer_idx)
                return _replace_hidden(output, h_new)
            return _hook

        def make_pre(layer_idx: int):
            def _hook(module, inputs):
                h = inputs[0]
                h_new = _apply(h, layer_idx)
                return (h_new,) + tuple(inputs[1:])
            return _hook

        for idx, mod in enumerate(targets):
            if pre_hook:
                handles.append(mod.register_forward_pre_hook(make_pre(idx)))
            else:
                handles.append(mod.register_forward_hook(make_post(idx)))

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()

        diag = {
            "htwr_hook_point": self.config.hook_point,
            "htwr_eta_effective": eta,
            "htwr_inject_layers": (
                "all" if self.config.inject_layers is None
                else list(int(i) for i in self.config.inject_layers)
            ),
            "htwr_inject_only_last_token": only_last,
        }
        return out, diag


# ---------------------------------------------------------------------------
# High-level helper: capture a bank from a list of write prompts
# ---------------------------------------------------------------------------


@dataclass
class WritePrompt:
    fact_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    subject: str | None = None
    relation: str | None = None
    target: str | None = None
    subject_token_index: int | None = None  # if known


def build_bank(
    injector: HTWRInjector,
    write_prompts: list[WritePrompt],
    *,
    capture_subject_keys: bool = False,
) -> HTWRMemoryBank:
    """Capture per-layer memories (and optional subject-token keys) for each fact."""
    if not write_prompts:
        raise ValueError("write_prompts is empty.")

    memories = []
    keys: list[torch.Tensor] = []
    subjects, relations, targets, fact_ids = [], [], [], []
    for wp in write_prompts:
        mem = injector.capture(wp.input_ids, wp.attention_mask)
        memories.append(mem)
        if capture_subject_keys and wp.subject_token_index is not None:
            sub_key = injector.capture(
                wp.input_ids, wp.attention_mask, token_index=int(wp.subject_token_index)
            )
            keys.append(sub_key)
        elif capture_subject_keys:
            keys.append(mem)  # fallback
        fact_ids.append(wp.fact_id)
        subjects.append(wp.subject)
        relations.append(wp.relation)
        targets.append(wp.target)

    memories_t = torch.stack(memories, dim=0)
    keys_t = torch.stack(keys, dim=0) if keys else None
    return HTWRMemoryBank(
        memories=memories_t,
        keys=keys_t,
        fact_ids=fact_ids,
        subjects=[s for s in subjects if s is not None] or None,
        relations=[r for r in relations if r is not None] or None,
        targets=[t for t in targets if t is not None] or None,
        hook_point=injector.config.hook_point,
        key_type="subject_token" if capture_subject_keys else "last_token",
    )


__all__ = [
    "HTWRConfig",
    "HTWRHookPoint",
    "HTWRMemoryBank",
    "HTWRInjector",
    "WritePrompt",
    "build_bank",
    "RetrievalResult",
    "HTWRRetriever",
    "OracleRetriever",
    "RandomRetriever",
    "RawCosineRetriever",
    "WhitenedCosineRetriever",
]
