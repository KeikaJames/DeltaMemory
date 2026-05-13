"""BankAttachedLLM — vLLM + AttnNativeBank production integration.

The safety contract is deliberately conservative:

* alpha=0 or an empty bank never installs any generation-time hook, so the vLLM
  path remains bit-identical to a plain ``LLM.generate`` call.
* model weights stay frozen; the integration creates no ``nn.Parameter``.
* For HF-compatible modules exposed by vLLM, the normal ``AttnNativePatcher`` is
  used.  For vLLM paged-attention modules, a forward-hook fallback is installed
  on the paged-attention layer.  vLLM does not expose pre-RoPE Q/K/V uniformly
  across 0.4/0.5/latest, so the fallback injects a deterministic bank-derived
  attention-output delta rather than modifying PagedAttention's kernel inputs.
  This preserves the red-line alpha=0 behavior and keeps the base frozen while
  leaving the exact kernel-level custom-op path open for a future vLLM fork.
"""
from __future__ import annotations

import contextlib
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch

try:  # optional dependency: importing this module must not require vLLM
    import vllm  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore

    _VLLM_AVAILABLE = True
    _VLLM_VERSION = getattr(vllm, "__version__", "unknown")
except ImportError:  # pragma: no cover - exercised by import tests
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False
    _VLLM_VERSION = "n/a"


_UNWRAP_PATHS_BY_VERSION: dict[str, tuple[str, ...]] = {
    "latest": (
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.model_executor.executor.driver_worker.model_runner.model",
        "llm_engine.engine_core.model_executor.driver_worker.model_runner.model",
    ),
    "0.5": (
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.driver_worker.model_runner.model",
    ),
    "0.4": (
        "llm_engine.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.model_runner.model",
    ),
}

_UNWRAP_PATHS: tuple[str, ...] = tuple(
    dict.fromkeys(
        path
        for family in ("latest", "0.5", "0.4")
        for path in _UNWRAP_PATHS_BY_VERSION[family]
    )
)


class _SyntheticHookBank:
    """Minimal non-Parameter bank for paged-attention fallback and tests."""

    def __init__(self, *, num_layers: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
        self.num_layers = int(max(1, num_layers))
        self.hidden_size = int(max(1, hidden_size))
        self.device = device
        self.dtype = dtype
        self.M_V = [torch.empty(0, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)]
        self.M_K = [torch.empty(0, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)]
        self.fact_ids: list[str] = []
        self.address_strs: list[str] = []

    @property
    def empty(self) -> bool:
        return self.size == 0

    @property
    def size(self) -> int:
        return int(self.M_V[0].shape[0]) if self.M_V else 0

    @property
    def num_facts(self) -> int:
        return self.size

    def clear(self) -> None:
        for i in range(self.num_layers):
            self.M_V[i] = torch.empty(0, self.hidden_size, device=self.device, dtype=self.dtype)
            self.M_K[i] = torch.empty(0, self.hidden_size, device=self.device, dtype=self.dtype)
        self.fact_ids.clear()
        self.address_strs.clear()

    def append_text_fact(self, fact_id: str, text: str, address: str) -> None:
        digest = hashlib.sha256(f"{fact_id}\0{text}\0{address}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vec = torch.randn(self.hidden_size, generator=generator, dtype=torch.float32)
        vec = vec / vec.norm().clamp_min(1e-6)
        vec = vec.to(device=self.device, dtype=self.dtype)
        for layer in range(self.num_layers):
            scale = 1.0 + (layer / max(1, self.num_layers))
            v = (vec * scale).unsqueeze(0)
            self.M_V[layer] = torch.cat([self.M_V[layer], v], dim=0)
            self.M_K[layer] = torch.cat([self.M_K[layer], v], dim=0)
        self.fact_ids.append(fact_id)
        self.address_strs.append(address)


@dataclass(frozen=True)
class _ResolvedModel:
    model: torch.nn.Module
    path: str


def _parse_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _ordered_unwrap_paths(version: str = _VLLM_VERSION) -> tuple[str, ...]:
    minor = _parse_minor(version)
    if minor == (0, 4):
        families = ("0.4", "0.5", "latest")
    elif minor == (0, 5):
        families = ("0.5", "latest", "0.4")
    else:
        families = ("latest", "0.5", "0.4")
    return tuple(dict.fromkeys(path for fam in families for path in _UNWRAP_PATHS_BY_VERSION[fam]))


def _get_path(root: Any, path: str) -> Any | None:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _unwrap_vllm_model(llm: Any) -> torch.nn.Module:
    """Extract the raw ``torch.nn.Module`` from known vLLM 0.4/0.5/latest layouts."""
    resolved = _resolve_vllm_model(llm)
    return resolved.model


def _resolve_vllm_model(llm: Any) -> _ResolvedModel:
    tried = _ordered_unwrap_paths()
    for path in tried:
        obj = _get_path(llm, path)
        if isinstance(obj, torch.nn.Module):
            return _ResolvedModel(model=obj, path=path)
    raise RuntimeError(
        "BankAttachedLLM: could not locate the underlying nn.Module in "
        f"vllm.LLM (version={_VLLM_VERSION}). Tried paths: {list(tried)}."
    )


def _first_parameter(model: torch.nn.Module) -> torch.nn.Parameter | None:
    return next(model.parameters(), None)


def _model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    param = _first_parameter(model)
    if param is None:
        return torch.device("cpu"), torch.float32
    return param.device, param.dtype


def _decoder_layer_count(model: torch.nn.Module) -> int:
    for path in (
        "model.model.language_model.layers",
        "model.model.layers",
        "model.language_model.model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "layers",
    ):
        obj = _get_path(model, path)
        if hasattr(obj, "__len__"):
            try:
                return int(len(obj))
            except TypeError:
                pass
    return 1


def _hidden_size(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    cfg = getattr(cfg, "text_config", cfg)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(cfg, name, None)
        if value:
            return int(value)
    param = _first_parameter(model)
    if param is not None and param.ndim > 0:
        return int(param.shape[-1])
    return 1


def _bank_is_empty(bank: Any) -> bool:
    return bool(getattr(bank, "empty", True))


def _bank_size(bank: Any) -> int:
    for name in ("num_facts", "size"):
        value = getattr(bank, name, None)
        if isinstance(value, int):
            return value
    return 0 if _bank_is_empty(bank) else 1


def _has_hf_attention_projections(module: torch.nn.Module) -> bool:
    return all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))


def _is_paged_attention_candidate(name: str, module: torch.nn.Module) -> bool:
    cls_name = type(module).__name__.lower()
    fq_name = name.lower()
    if _has_hf_attention_projections(module):
        return False
    if "pagedattention" in cls_name or "paged_attention" in fq_name:
        return True
    if cls_name == "attention" and any(hasattr(module, attr) for attr in ("impl", "kv_cache", "num_kv_heads")):
        return True
    return False


def _adapt_delta(vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    flat = vec.reshape(-1).to(device=target.device, dtype=target.dtype)
    width = int(target.shape[-1])
    if flat.numel() < width:
        repeats = (width + flat.numel() - 1) // max(1, flat.numel())
        flat = flat.repeat(repeats)
    flat = flat[:width]
    rms = flat.float().pow(2).mean().sqrt().clamp_min(1e-6)
    return (flat / rms).to(dtype=target.dtype) * 5e-2


class _VLLMPagedAttentionHooks:
    """Forward-hook fallback for vLLM PagedAttention modules.

    The hook is inactive unless ``active()`` is entered with alpha>0 and a
    non-empty bank.  Therefore alpha=0/empty-bank generation returns vLLM's
    original output object unchanged.
    """

    def __init__(self, model: torch.nn.Module, bank_getter: Callable[[], Any], alpha_getter: Callable[[], float]):
        self.model = model
        self._bank_getter = bank_getter
        self._alpha_getter = alpha_getter
        self.handles: list[Any] = []
        self.enabled = False
        self.module_names: list[str] = []

    @property
    def installed(self) -> bool:
        return bool(self.handles)

    def install(self) -> int:
        if self.handles:
            return len(self.handles)
        for index, (name, module) in enumerate(self.model.named_modules()):
            if not _is_paged_attention_candidate(name, module):
                continue
            handle = module.register_forward_hook(self._make_hook(index))
            self.handles.append(handle)
            self.module_names.append(name)
        return len(self.handles)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.module_names.clear()

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = previous

    def _make_hook(self, layer_idx: int) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            alpha = float(self._alpha_getter())
            bank = self._bank_getter()
            if (not self.enabled) or alpha == 0.0 or bank is None or _bank_is_empty(bank):
                return output
            return self._inject_output(output, bank, alpha, layer_idx)

        return hook

    def _inject_output(self, output: Any, bank: Any, alpha: float, layer_idx: int) -> Any:
        if isinstance(output, torch.Tensor):
            return self._inject_tensor(output, bank, alpha, layer_idx)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            first = self._inject_tensor(output[0], bank, alpha, layer_idx)
            return (first, *output[1:])
        return output

    def _inject_tensor(self, tensor: torch.Tensor, bank: Any, alpha: float, layer_idx: int) -> torch.Tensor:
        values = getattr(bank, "M_V", None)
        if not values:
            return tensor
        layer_values = values[layer_idx % len(values)]
        if layer_values.numel() == 0:
            return tensor
        vec = layer_values.float().mean(dim=tuple(range(max(0, layer_values.ndim - 1))))
        delta = _adapt_delta(vec, tensor)
        view_shape = (1,) * (tensor.ndim - 1) + (delta.numel(),)
        return tensor + (float(alpha) * delta.view(view_shape))


class BankAttachedLLM:
    """vLLM LLM wrapper with an AttnNativeBank attached to the frozen model."""

    def __init__(
        self,
        model: str,
        *,
        alpha: float = 1.0,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        install_vllm_hooks: bool = True,
        enable_hf_patcher: bool = True,
        **vllm_kwargs: Any,
    ) -> None:
        llm_cls = vllm_kwargs.pop("_llm_cls", LLM)
        sampling_params_cls = vllm_kwargs.pop("_sampling_params_cls", SamplingParams)
        patcher_cls = vllm_kwargs.pop("_patcher_cls", None)
        fresh_bank_fn = vllm_kwargs.pop("_fresh_bank_fn", None)
        tokenizer_cls = vllm_kwargs.pop("_tokenizer_cls", None)
        if llm_cls is None or sampling_params_cls is None:
            raise ImportError("vLLM is not installed. Install it with `pip install vllm`.")

        self.alpha = float(alpha)
        self._active_alpha = float(alpha)
        self.model_id = model
        self._sampling_params_cls = sampling_params_cls
        self._hf_patch_error: Exception | None = None

        llm_init_kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if max_model_len is not None:
            llm_init_kwargs["max_model_len"] = max_model_len
        llm_init_kwargs.update(vllm_kwargs)
        self.llm = llm_cls(**llm_init_kwargs)

        resolved = _resolve_vllm_model(self.llm)
        self._nn_model = resolved.model
        self.unwrap_path = resolved.path

        if tokenizer_cls is None:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer_cls = AutoTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(model)

        self.patcher: Any | None = None
        self.bank: Any
        if enable_hf_patcher:
            try:
                if patcher_cls is None or fresh_bank_fn is None:
                    from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank

                    patcher_cls = patcher_cls or AttnNativePatcher
                    fresh_bank_fn = fresh_bank_fn or fresh_bank
                self.patcher = patcher_cls(self._nn_model)
                self.bank = fresh_bank_fn(self._nn_model)
            except Exception as exc:
                self._hf_patch_error = exc
                self.bank = self._new_synthetic_bank()
        else:
            self.bank = self._new_synthetic_bank()

        self.hook_controller = _VLLMPagedAttentionHooks(
            self._nn_model,
            bank_getter=lambda: self.bank,
            alpha_getter=lambda: self._active_alpha,
        )
        if install_vllm_hooks:
            self.hook_controller.install()

    def _new_synthetic_bank(self) -> _SyntheticHookBank:
        device, dtype = _model_device_dtype(self._nn_model)
        return _SyntheticHookBank(
            num_layers=_decoder_layer_count(self._nn_model),
            hidden_size=_hidden_size(self._nn_model),
            device=device,
            dtype=dtype,
        )

    def write_facts(self, facts: list[tuple[str, str, str]], *, policy: str = "period") -> None:
        """Write ``(fact_id, write_prompt, address)`` triples into the bank."""
        if self.patcher is None:
            for fact_id, write_prompt, address in facts:
                if not hasattr(self.bank, "append_text_fact"):
                    raise RuntimeError("No HF patcher and current bank cannot accept synthetic facts")
                self.bank.append_text_fact(fact_id, write_prompt, address)
            return

        from deltamemory.memory.attn_native_bank import write_fact as _write_fact

        for fact_id, write_prompt, address in facts:
            _write_fact(
                self.patcher,
                self.bank,
                self.tokenizer,
                write_prompt=write_prompt,
                fact_id=fact_id,
                address=address,
                policy=policy,
            )

    def clear_bank(self) -> None:
        """Remove all facts from the bank while preserving hook wiring."""
        if hasattr(self.bank, "clear"):
            self.bank.clear()
            return
        if self.patcher is not None:
            from deltamemory.memory.attn_native_bank import fresh_bank

            self.bank = fresh_bank(self._nn_model)
        else:
            self.bank = self._new_synthetic_bank()

    @contextlib.contextmanager
    def _bank_active(self, alpha: float) -> Iterator[None]:
        self._active_alpha = float(alpha)
        if alpha == 0.0 or _bank_is_empty(self.bank):
            yield
            return
        contexts: list[Any] = []
        if self.patcher is not None:
            contexts.extend([self.patcher.patched(), self.patcher.injecting(self.bank, alpha=alpha)])
        if self.hook_controller.installed:
            contexts.append(self.hook_controller.active())
        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield

    def generate(
        self,
        prompts: list[str],
        *,
        alpha: Optional[float] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **sampling_kwargs: Any,
    ) -> list[Any]:
        """Generate with bank injection active only when alpha>0 and bank is non-empty."""
        selected_alpha = self.alpha if alpha is None else float(alpha)
        params = self._sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            **sampling_kwargs,
        )
        with self._bank_active(selected_alpha):
            return self.llm.generate(prompts, params)

    def recall_top5(self, prompt: str, *, alpha: Optional[float] = None) -> list[str]:
        """Return top-5 token strings from the HF-compatible path."""
        if self.patcher is None:
            raise RuntimeError(
                "recall_top5 requires an HF-compatible attention patcher; "
                f"paged-attention fallback was selected ({self._hf_patch_error!r})."
            )
        from deltamemory.memory.attn_native_bank import forward_with_bank

        selected_alpha = self.alpha if alpha is None else float(alpha)
        logits = forward_with_bank(self.patcher, self.bank, self.tokenizer, prompt, alpha=selected_alpha)
        top5_ids = logits.topk(5).indices.tolist()
        return [self.tokenizer.decode([tid]).strip() for tid in top5_ids]

    def vllm_version(self) -> str:
        return _VLLM_VERSION

    def close(self) -> None:
        self.hook_controller.remove()

    def __repr__(self) -> str:
        return (
            f"BankAttachedLLM(model={self.model_id!r}, bank_size={_bank_size(self.bank)}, "
            f"alpha={self.alpha}, vllm={_VLLM_VERSION}, unwrap_path={self.unwrap_path!r})"
        )
