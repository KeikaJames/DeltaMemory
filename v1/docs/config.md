# Config Schema Reference

This document describes the pydantic v2 config schemas added in **v0.5 Phase I.3**.
These schemas are **opt-in** — no existing injector is modified.  Migration is a
separate future PR.  Existing dataclasses coexist with these schemas unchanged.

---

## Overview

All schemas live in `deltamemory/config/`.

| Import | Purpose |
|--------|---------|
| `BaseInjectorConfig` | Shared base for all injector configs |
| `LopiConfig` | LOPI injector |
| `CaaConfig` | CAA injector |
| `ScarConfig` | SCAR injector |
| `AttnNativeBankConfig` | Attention Native Bank injector |
| `RomeWriterConfig` | ROME writer injector |
| `MnemeWriterConfig` | Mneme writer injector |
| `ServiceConfig` | FastAPI / Uvicorn runtime |
| `load_config(path)` | Parse YAML → validated models |
| `dump_config(cfg, path)` | Serialise models → YAML |

---

## Schema Reference

### `BaseInjectorConfig`

Shared fields inherited by every injector schema.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `alpha` | `float` | — | ≥ 0 | Injection strength |
| `layer_idx` | `int` | `-1` | ≥ −1 | Target layer index; −1 = auto-select |
| `enabled` | `bool` | `True` | — | Whether the injector is active |
| `dtype` | `"fp32"│"bf16"│"fp16"` | `"bf16"` | — | Compute dtype |
| `device` | `"cpu"│"cuda"│"mps"` | `"cpu"` | — | Target device |

---

### `LopiConfig`

Inherits `BaseInjectorConfig`.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `eta_sigma` | `float` | — | (0, 2] | Noise scale σ |
| `gate_k` | `float` | — | > 0 | Sigmoid gate steepness |
| `gate_theta` | `float` | — | ≥ 0 | Sigmoid gate shift |
| `use_derivative_gate` | `bool` | `True` | — | Enable derivative-based gating |

---

### `CaaConfig`

Inherits `BaseInjectorConfig`.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `n_pairs` | `int` | — | ≥ 1 | Number of contrastive pairs |
| `use_lopi_gate` | `bool` | `False` | — | Route through LOPI gate |
| `gate_k` | `float` | — | > 0 | Sigmoid gate steepness |
| `gate_theta` | `float` | — | ≥ 0 | Sigmoid gate shift |

---

### `ScarConfig`

Inherits `BaseInjectorConfig`.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `projection` | `"m_perp"│"raw"` | `"m_perp"` | — | Projection mode |
| `subspace_rank` | `int` | — | ≥ 1 | Rank of the retained subspace |

---

### `AttnNativeBankConfig`

Inherits `BaseInjectorConfig`.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `bank_size` | `int` | — | ≥ 1 | Maximum stored attention vectors |
| `top_k` | `int` | — | ≥ 1, ≤ `bank_size` | Top matches to retrieve |
| `theta` | `float` | — | [−π, π] | Angular threshold (radians) |
| `capture` | `"pre_rope"│"post_rope"` | `"pre_rope"` | — | Key capture timing |

> **Cross-field invariant**: `top_k` ≤ `bank_size` is enforced via a `model_validator`.

---

### `RomeWriterConfig`

Inherits `BaseInjectorConfig`.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `n_optim_steps` | `int` | — | ≥ 1 | Number of optimisation steps |

---

### `MnemeWriterConfig`

Inherits `BaseInjectorConfig`.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `write_alpha` | `float` | — | ≥ 0 | Write-path scaling factor |

---

### `ServiceConfig`

FastAPI / Uvicorn runtime settings.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `bind_host` | `str` | — | — | Host address |
| `port` | `int` | — | [1, 65535] | TCP port |
| `workers` | `int` | — | ≥ 1 | Uvicorn worker count |
| `request_size_cap_mb` | `int` | — | ≥ 1 | Max request body (MiB) |
| `jwt_required` | `bool` | `True` | — | Require JWT on every request |
| `prometheus_enabled` | `bool` | `True` | — | Expose `/metrics` endpoint |

---

## YAML Example Walkthrough

The canonical example lives at `deltamemory/config/example.yaml`.

```yaml
injectors:

  lopi:
    alpha: 0.5          # injection strength  (≥ 0)
    layer_idx: 16       # target layer; -1 = auto
    enabled: true
    dtype: bf16         # fp32 | bf16 | fp16
    device: cuda
    eta_sigma: 0.8      # noise σ  ∈ (0, 2]
    gate_k: 2.0         # steepness > 0
    gate_theta: 0.1     # shift ≥ 0
    use_derivative_gate: true

  attn_native_bank:
    alpha: 0.7
    bank_size: 256
    top_k: 16           # ≤ bank_size  ← cross-field invariant
    theta: 0.523599     # ≈ π/6  ∈ [-π, π]
    capture: pre_rope

service:
  bind_host: "0.0.0.0"
  port: 8080
  workers: 4
  request_size_cap_mb: 16
  jwt_required: true
  prometheus_enabled: true
```

### Loading in Python

```python
from deltamemory.config import load_config, LopiConfig

cfg = load_config("deltamemory/config/example.yaml")
lopi: LopiConfig = cfg["injectors.lopi"]
print(lopi.alpha, lopi.eta_sigma)
```

### Round-trip serialisation

```python
from deltamemory.config import load_config, dump_config

cfg = load_config("example.yaml")
# mutate
cfg["service"].workers = 8
dump_config(cfg, "example_modified.yaml")
```

---

## Validation Errors

Pydantic raises `ValidationError` on constraint violations.  Examples:

```python
from pydantic import ValidationError
from deltamemory.config import AttnNativeBankConfig

try:
    AttnNativeBankConfig(alpha=0.5, bank_size=4, top_k=8, theta=0.0, capture="pre_rope")
except ValidationError as e:
    print(e)
# Value error: top_k (8) must not exceed bank_size (4)
```
