"""Phase R-6 — persistent bank storage for ``AttnNativeBank``.

Industrial-scale bank storage with three guarantees:

1. **Versioned, content-addressed directories**: every save lands at
   ``<root>/<model_safe>/<config_sha>/`` where ``config_sha`` is a sha256 of
   the bank-relevant config (architecture shape + LOPI cfg + bank temperature
   + shield flag).  Different model / config combinations cannot collide.

2. **safetensors per-fact tensors**: ``M_K`` / ``M_V`` per layer are written
   as a single zero-copy mmap-able ``bank.safetensors`` file.  Pickle-free,
   memory-safe, cross-language.

3. **File-locked writes**: concurrent writers to the same ``config_sha`` are
   serialised through ``filelock``; readers see only fully-written snapshots
   thanks to atomic ``os.replace``.

The persisted bank can be reloaded into a fresh ``AttnNativeBank`` whose
``state_dict()`` round-trips bit-equal under the same dtype.

This module intentionally **does not** add FAISS; current bank sizes
(<= 1k facts) are well below the FAISS break-even.  ANN indexing is queued
for R-6.2 once bank sizes pass 10k facts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - hard dep, tested in CI
    FileLock = None  # type: ignore

VERSION = "ulopi_v36"
META_FILENAME = "meta.json"
TENSORS_FILENAME = "bank.safetensors"
LOCK_FILENAME = ".lock"
# v3.4/v3.5 banks remain readable: see _LEGACY_VERSIONS in load_bank.
_LEGACY_VERSIONS = ("lopi_v33", "ulopi_v35")


# ---------------------------------------------------------------------------
# Config sha
# ---------------------------------------------------------------------------

def _model_safe(model_name: str) -> str:
    """Return a filesystem-safe slug for a HF model id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def compute_config_sha(
    *,
    model_name: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    head_dims: list[int] | None,
    dtype: str,
    bank_temperature: float = 1.0,
    mhc_shield: bool = False,
    lopi_cfg: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable sha256 over the bank-relevant configuration.

    Two banks share a ``config_sha`` iff they are mutually loadable without
    schema migration: same arch shape, same dtype, same LOPI defaults.
    """
    payload = {
        "version": VERSION,
        "model_name": model_name,
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "head_dims": list(head_dims) if head_dims else None,
        "dtype": str(dtype),
        "bank_temperature": float(bank_temperature),
        "mhc_shield": bool(mhc_shield),
        "lopi_cfg": dict(lopi_cfg) if lopi_cfg else None,
        "extra": dict(extra) if extra else None,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class BankLocation:
    """Resolved on-disk location for a (model, config) pair."""
    root: Path
    model_safe: str
    config_sha: str

    @property
    def dir(self) -> Path:
        return self.root / self.model_safe / self.config_sha

    @property
    def meta_path(self) -> Path:
        return self.dir / META_FILENAME

    @property
    def tensors_path(self) -> Path:
        return self.dir / TENSORS_FILENAME

    @property
    def lock_path(self) -> Path:
        return self.dir / LOCK_FILENAME


def resolve_location(
    root: str | Path,
    model_name: str,
    config_sha: str,
) -> BankLocation:
    return BankLocation(
        root=Path(root),
        model_safe=_model_safe(model_name),
        config_sha=config_sha,
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _lopi_cfg_to_dict(cfg: Any) -> dict[str, Any] | None:
    if cfg is None:
        return None
    if hasattr(cfg, "asdict"):
        return _jsonable(cfg.asdict())
    keys = ("enabled", "orthogonal", "gaussian", "derivative",
            "profile_mode", "z_clamp", "auto_mu_c",
            "norm_base", "k_shift", "theta_shift", "k_gate", "theta_gate",
            "kappa_depth", "beta_sigma", "mu_lo", "mu_hi", "mu_low", "mu_span",
            "sigma_floor", "epsilon", "eps", "use_ecor", "ecor_cfg")
    out: dict[str, Any] = {}
    for k in keys:
        if hasattr(cfg, k):
            out[k] = _jsonable(getattr(cfg, k))
    return out


def _lopi_cfg_from_dict(cfg_dict: dict[str, Any] | None) -> Any:
    if not cfg_dict:
        return None
    from deltamemory.memory.lopi import LOPIConfig
    from deltamemory.memory.lopi_inject import ECORConfig

    cfg_fields = {f.name for f in fields(LOPIConfig)}
    kwargs = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    ecor_dict = kwargs.get("ecor_cfg")
    if isinstance(ecor_dict, dict):
        ecor_fields = {f.name for f in fields(ECORConfig)}
        kwargs["ecor_cfg"] = ECORConfig(**{k: v for k, v in ecor_dict.items() if k in ecor_fields})
    return LOPIConfig(**kwargs)


def _lopi_profile_to_dict(state: Any) -> dict[str, Any] | None:
    """Phase S — extract LOPIProfile (if any) from a bank's lopi_state."""
    if state is None:
        return None
    profile = getattr(state, "profile", None)
    if profile is None:
        return None
    if hasattr(profile, "asdict"):
        return profile.asdict()
    return dict(profile)  # type: ignore[arg-type]


def _bank_runtime_cfg(sd: dict[str, Any]) -> dict[str, Any]:
    """Runtime attention knobs that affect read semantics but live on bank attrs."""
    return {
        "bank_cosine": bool(sd.get("bank_cosine", False)),
        "bank_topk": int(sd.get("bank_topk", 0) or 0),
        "bank_separate_softmax": bool(sd.get("bank_separate_softmax", False)),
        "bank_merge_beta": float(sd.get("bank_merge_beta", 1.0)),
    }


def save_bank(
    bank: Any,
    root: str | Path,
    *,
    model_name: str,
    extra_meta: dict[str, Any] | None = None,
) -> BankLocation:
    """Persist ``bank`` (an ``AttnNativeBank``) to ``root`` under a content-addressed dir.

    Returns the resolved :class:`BankLocation`.  After this call the directory
    contains ``bank.safetensors`` and ``meta.json``; concurrent writers are
    serialised on ``.lock``.
    """
    sd = bank.state_dict()
    # dtype from first non-empty tensor; fall back to bfloat16
    dtype_str = "bfloat16"
    for t in sd["M_K"]:
        if t.numel() > 0:
            dtype_str = str(t.dtype).replace("torch.", "")
            break

    profile_dict = _lopi_profile_to_dict(getattr(bank, "lopi_state", None))
    profile_corpus_sha = profile_dict.get("profile_corpus_sha") if profile_dict else None

    config_sha = compute_config_sha(
        model_name=model_name,
        num_layers=sd["num_layers"],
        num_kv_heads=sd["num_kv_heads"],
        head_dim=sd["head_dim"],
        head_dims=sd.get("head_dims"),
        dtype=dtype_str,
        bank_temperature=sd.get("bank_temperature", 1.0),
        mhc_shield=sd.get("mhc_shield", False),
        lopi_cfg=_lopi_cfg_to_dict(getattr(bank, "lopi_cfg", None)),
        extra={
            "profile_corpus_sha": profile_corpus_sha,
            "value_scale_mode": sd.get("value_scale_mode", "auto_rms_cap"),
            "value_target_rms": float(sd.get("value_target_rms", 0.5)),
            **_bank_runtime_cfg(sd),
        },
    )
    loc = resolve_location(root, model_name, config_sha)
    loc.dir.mkdir(parents=True, exist_ok=True)

    if FileLock is None:
        raise RuntimeError("filelock is required for concurrent-safe writes")

    with FileLock(str(loc.lock_path), timeout=60):
        # Flatten per-layer tensors: M_K_<layer>, M_V_<layer>
        tensors: dict[str, torch.Tensor] = {}
        for layer, (k, v) in enumerate(zip(sd["M_K"], sd["M_V"])):
            tensors[f"M_K_{layer}"] = k.contiguous().to("cpu")
            tensors[f"M_V_{layer}"] = v.contiguous().to("cpu")

        meta = {
            "version": VERSION,
            "model_name": model_name,
            "config_sha": config_sha,
            "saved_at_unix": time.time(),
            "num_layers": int(sd["num_layers"]),
            "num_kv_heads": int(sd["num_kv_heads"]),
            "head_dim": int(sd["head_dim"]),
            "head_dims": list(sd.get("head_dims") or []),
            "dtype": dtype_str,
            "bank_temperature": float(sd.get("bank_temperature", 1.0)),
            "mhc_shield": bool(sd.get("mhc_shield", False)),
            "value_scale_mode": str(sd.get("value_scale_mode", "auto_rms_cap")),
            "value_target_rms": float(sd.get("value_target_rms", 0.5)),
            "value_scale_eps": float(sd.get("value_scale_eps", 1e-6)),
            "lopi_cfg": _lopi_cfg_to_dict(getattr(bank, "lopi_cfg", None)),
            "lopi_profile": profile_dict,
            **_bank_runtime_cfg(sd),
            "n_facts": int(sd["M_K"][0].size(0)) if sd["M_K"] else 0,
            "fact_ids": list(sd.get("fact_ids", [])),
            "address_strs": list(sd.get("address_strs", [])),
            "extra": dict(extra_meta) if extra_meta else None,
        }

        # Atomic write: tmp + rename
        tmp_tensors = loc.tensors_path.with_suffix(".tmp")
        tmp_meta = loc.meta_path.with_suffix(".tmp")
        save_file(tensors, str(tmp_tensors))
        with tmp_meta.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp_tensors, loc.tensors_path)
        os.replace(tmp_meta, loc.meta_path)

    return loc


def load_bank(
    location: BankLocation | tuple[str | Path, str, str],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Any:
    """Reload a bank previously saved by :func:`save_bank`.

    ``location`` is either a :class:`BankLocation` or a tuple
    ``(root, model_name, config_sha)``.

    Returns an ``AttnNativeBank`` on the requested device/dtype.  When
    ``dtype`` is None the original on-disk dtype is preserved.
    """
    from deltamemory.memory.attn_native_bank import AttnNativeBank

    if not isinstance(location, BankLocation):
        root, model_name, config_sha = location
        location = resolve_location(root, model_name, config_sha)

    with location.meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    on_disk_version = meta.get("version")
    if on_disk_version != VERSION and on_disk_version not in _LEGACY_VERSIONS:
        raise ValueError(
            f"bank schema version mismatch: on-disk={on_disk_version!r} "
            f"runtime={VERSION!r}; migration not implemented"
        )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    meta_dtype = str(meta["dtype"])
    if meta_dtype not in dtype_map:
        raise ValueError(f"unsupported bank dtype in metadata: {meta_dtype!r}")
    on_disk_dtype = dtype_map[meta_dtype]
    target_dtype = dtype or on_disk_dtype

    # Read tensors
    flat = load_file(str(location.tensors_path), device="cpu")
    n_layers = int(meta["num_layers"])
    M_K = [flat[f"M_K_{i}"] for i in range(n_layers)]
    M_V = [flat[f"M_V_{i}"] for i in range(n_layers)]

    sd = {
        "num_layers": n_layers,
        "num_kv_heads": int(meta["num_kv_heads"]),
        "head_dim": int(meta["head_dim"]),
        "head_dims": list(meta.get("head_dims") or []),
        "M_K": M_K,
        "M_V": M_V,
        "fact_ids": list(meta.get("fact_ids", [])),
        "address_strs": list(meta.get("address_strs", [])),
        "bank_temperature": float(meta.get("bank_temperature", 1.0)),
        "mhc_shield": bool(meta.get("mhc_shield", False)),
        "value_scale_mode": str(meta.get("value_scale_mode", "auto_rms_cap")),
        "value_target_rms": float(meta.get("value_target_rms", 0.5)),
        "value_scale_eps": float(meta.get("value_scale_eps", 1e-6)),
        "bank_cosine": bool(meta.get("bank_cosine", False)),
        "bank_topk": int(meta.get("bank_topk", 0) or 0),
        "bank_separate_softmax": bool(meta.get("bank_separate_softmax", False)),
        "bank_merge_beta": float(meta.get("bank_merge_beta", 1.0)),
    }
    bank = AttnNativeBank.from_state_dict(sd, device=device, dtype=target_dtype)
    lopi_cfg = _lopi_cfg_from_dict(meta.get("lopi_cfg"))
    if lopi_cfg is not None:
        bank.lopi_cfg = lopi_cfg

    # Phase S — restore the LOPI profile (if any).  We attach it directly to
    # the bank's freshly-constructed lopi_state so reloads inherit the
    # auto-calibration without needing to re-profile the model.
    profile_dict = meta.get("lopi_profile")
    if profile_dict:
        from deltamemory.memory.lopi_profiler import LOPIProfile
        bank.lopi_state.profile = LOPIProfile.from_dict(profile_dict)
    return bank


def storage_bytes(location: BankLocation) -> int:
    """Total disk footprint of a saved bank directory."""
    if not location.dir.exists():
        return 0
    return sum(p.stat().st_size for p in location.dir.rglob("*") if p.is_file())


def list_banks(root: str | Path, model_name: str | None = None) -> list[BankLocation]:
    """Enumerate persisted banks under ``root``."""
    root = Path(root)
    if not root.exists():
        return []
    out: list[BankLocation] = []
    model_dirs = (
        [root / _model_safe(model_name)] if model_name else
        [d for d in root.iterdir() if d.is_dir()]
    )
    for mdir in model_dirs:
        if not mdir.exists():
            continue
        for cfg_dir in mdir.iterdir():
            if cfg_dir.is_dir() and (cfg_dir / META_FILENAME).exists():
                out.append(BankLocation(
                    root=root,
                    model_safe=mdir.name,
                    config_sha=cfg_dir.name,
                ))
    return out
