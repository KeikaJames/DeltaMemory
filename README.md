> [!WARNING]
> This repository is a research prototype involving LLM hidden-state,
> attention-layer, tensor-bank, and injection mechanisms.
>
> Use of this repository is accompanied by the Security Policy and Responsible
> Use Protocol set out in [`MnEmE/docs/security.md`](./docs/security.md).
>
> Users, operators, distributors, and downstream recipients are responsible for
> complying with applicable laws, third-party rights, platform terms, and the
> security requirements described in that document.
<p align="center">
  <h1 align="center">Mneme</h1>
</p>

<p align="center">
  <strong>External K/V memory injected inside frozen Transformer attention.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-EE4C2C.svg">
  <img alt="Hardware" src="https://img.shields.io/badge/Apple%20MPS%20%7C%20GB10%20CUDA-bf16-555.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange.svg">
</p>

<p align="center">
  <strong>Languages:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="docs/design.md">Design</a> ·
  <a href="docs/apple_silicon.md">Apple Silicon</a> ·
  <a href="docs/HISTORY.md">Phase history</a>
</p>

---

Mneme is a research prototype for **persistent external memory in a
frozen LLM**. A per-layer K/V bank is concatenated into supported attention
layers; the prompt at read time contains only the question, and the base
weights stay frozen. The default production path is the attention-native bank
with architecture-specific α defaults and V-scale calibration; Dynamic LOPI /
U-LOPI and mHC are available as explicit ablation knobs, not hidden prompt
context. It is **not RAG**, **not prompt insertion**, and **not a weight edit**.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamemory import (
    AttnNativePatcher, fresh_bank, write_fact,
    LOPIConfig,
    save_bank, load_bank,
)

model_name = "google/gemma-4-E2B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

# 1) Bank + per-layer attention patcher around the frozen LLM.
patcher = AttnNativePatcher(model)
bank = fresh_bank(model)
bank.lopi_cfg = LOPIConfig(enabled=False)  # set True only for LOPI ablations

# 2) Optional U-LOPI cold-start for LOPI ablations.
# bank.lopi_cfg = LOPIConfig(enabled=True, profile_mode="auto")
# bank.attach_lopi_profile(model, tok)          # forward-only, weights bit-equal

# 3) Write a fact into the bank.
write_fact(patcher, bank, tok,
           write_prompt="Fact: Python was created by Ada Lovelace.",
           fact_id="py_ada", address="Python")

# 4) Decode with the bank attached.
read_prompt = "Q: Who created the Python programming language?\nA:"
with patcher.patched(), patcher.injecting(bank, alpha=1.0), torch.no_grad():
    out = model.generate(**tok(read_prompt, return_tensors="pt"), max_new_tokens=8)
print(tok.decode(out[0], skip_special_tokens=True))

# 5) Persist (schema "ulopi_v36"); load_bank restores LOPI/ECOR, profile,
#    V-scale, and bank-attention runtime config.
save_bank(bank, root="./banks", model_name=model_name)
```

`bank.attach_lopi_profile(...)` is a thin wrapper around `profile_residuals`
that binds the profile onto `bank.lopi_state` and validates the layer count
matches the bank shape. With `LOPIConfig(enabled=False)` (the default) the
merged-softmax branch is bit-for-bit equivalent to the legacy v3.1 formula,
and `α=0` / empty bank stays bit-equal to the unmodified model.

## Architecture

### AttnNativeBank

A frozen-LLM external attention bank. Per non-shared attention layer ℓ, the
bank stores pre-RoPE K and post-norm V tensors at the model's
`num_key_value_heads` resolution, concatenated into attention itself:

$$
\mathrm{Attn}_\ell\bigl(Q,\; [K\,;\, M_K^{(\ell)}],\; [V\,;\, \alpha M_V^{(\ell)}]\bigr)
$$

The bank carries no learnable parameters; the retrieval space *is* the
model's native K-space and the attention softmax is the contrastive engine.
GQA / MQA expansion uses the model's own `repeat_kv`. KV-shared layers
(e.g. Gemma 4) consult their source layer's bank slot at read time so every
attention layer sees the bank.

* File: [`deltamemory/memory/attn_native_bank.py`](deltamemory/memory/attn_native_bank.py)
* Patcher: `AttnNativePatcher`. Helpers: `fresh_bank`, `write_fact`, `forward_with_bank`.
* Bit-equal sanity: `tests/test_attn_native_bank.py`.

### Dynamic LOPI v3.4

A training-free wrapper at the merged-softmax branch that replaces

`out_bank = weights[..., T:] @ (alpha * mv_e)`

with three independent, config-switchable components:

$$
\mathrm{out\_bank}_{\mathrm{LOPI}} \;=\; \gamma_t \cdot w(\ell, t) \cdot M_\perp,
\quad M_\perp = M_V - \mathrm{proj}_{V_{\mathrm{ctx}}}(M_V)
$$

* **Orthogonal Novelty** (`M_perp`) drops the bank component parallel to the
  native context value. (v3.4 default: off; flip on for v3.3 ablations.)
* **Adaptive Layer Gaussian** `w(ℓ, t)` — Gaussian over layer index, centred
  at `μ_t` driven by the previous step's residual norm, width `σ_t` shrunk
  by a running mHC max-σ stability signal.
* **Derivative Gate** `γ_t = sigmoid(k · (‖Q_t − Q_{t-1}‖₂ − θ))` silences
  injection when the topic is stable and opens it during topic shifts.

`LOPIConfig(enabled=True, orthogonal=False, gaussian=True, derivative=True)`
is the v3.4 default; `enabled=False` and `α=0` are both bit-equal to the
unmodified model.

* File: [`deltamemory/memory/lopi.py`](deltamemory/memory/lopi.py)
* Public symbols: `LOPIConfig`, `LOPIState`, `apply_lopi`, `derivative_gate`,
  `layer_gaussian_weight`, `orthogonal_novelty`.

### U-LOPI Phase S

v3.4 hard-coded `norm_base = 10.0`, calibrated to Gemma-4-E2B and silently
degraded on other families whose residual-stream scale differs by 10–100×.
Phase S replaces the global constant with a one-shot cold-start profile of
`‖hidden_states[ℓ]‖₂` over a small neutral corpus, persisted alongside the
bank. The depth signal is then computed in Z-score space and `μ_t` is
auto-anchored at the architecture's spike layer:

$$
z_\ell(t) \;=\; \frac{N_t(\ell) - \mu_{\mathrm{base}}(\ell)}{\sigma_{\mathrm{base}}(\ell) + \varepsilon},
\qquad
\mu_{\mathrm{arch}} \;=\; \arg\max_\ell\, \sigma_{\mathrm{base}}(\ell)
$$

The forward is `output_hidden_states=True`-only — no `nn.Parameter` is
introduced and the LLM weights are bit-equal pre/post (verified by
`test_lopi_profiler.py::test_profile_does_not_mutate_weights`). With
`LOPIConfig(profile_mode="auto")` (default), `norm_base` / `mu_low` /
`mu_span` are ignored at runtime; `profile_mode="static"` reproduces v3.4
exactly for regression checks.

* File: [`deltamemory/memory/lopi_profiler.py`](deltamemory/memory/lopi_profiler.py)
* Public symbols: `LOPIProfile`, `profile_residuals`, `default_profile_corpus`,
  `save_profile`, `load_profile`.
* Cross-arch coverage: `tests/test_lopi_universal.py` (Gemma / Qwen3 / GLM-4
  / Llama / GPT-2 shape and bit-equality checks).

### Persistence (Phase R-6)

Versioned, content-addressed bank storage at
`<root>/<model_safe>/<config_sha>/`, where `config_sha` is sha256 of the
bank-relevant config (architecture shape + LOPI cfg + bank temperature +
shield flag + V-scale calibration). `M_K`/`M_V` per layer are written as a single zero-copy
mmap-able `bank.safetensors`; concurrent writes are serialised through
`filelock` and readers see only fully-written snapshots thanks to atomic
`os.replace`. The persisted snapshot includes the Phase S `LOPIProfile`, so
reloads inherit per-arch calibration. Format version: `ulopi_v36`.

$$
\mathrm{config\_sha} \;=\; \mathrm{sha256}\!\bigl(\,\mathrm{shape}\;\Vert\;\mathrm{LOPIConfig}\;\Vert\;\tau\;\Vert\;\mathrm{shield}\;\Vert\;\mathrm{VScale}\bigr)
$$

* File: [`deltamemory/memory/bank_persistence.py`](deltamemory/memory/bank_persistence.py)
* Public symbols: `save_bank`, `load_bank`, `list_banks`,
  `compute_config_sha`, `resolve_location`.
* Round-trip tests: `tests/test_bank_persistence.py`.

## Phase history

| Phase | What shipped | Evidence | Status |
|---|---|---|---|
| Stages 0–14 | v1 → v3 (writer / address bank / K-projector) | local archive | superseded; see [`docs/HISTORY.md`](docs/HISTORY.md) |
| Stage 15 / v3.1 | attn-native bank + per-arch α + cross-arch adapters | local archive | reproduced on Gemma-4 and Qwen3 (GB10/Mac) |
| Stage 16 / v3.2 | mHC spectral shield (column-cap on bank weights) | local archive | bounds σ_max(W) ≤ 1; α=0 bit-equality preserved |
| R-3 / v3.3 | Dynamic LOPI ablation (A0–A4, 630 cells) | local archive | preregistered cleanroom run |
| R-3.5 / v3.4 | default flip → `orthogonal=False, gaussian=True, derivative=True` | local archive | high-α drift collapse + α=1 lift preserved |
| R-4 / v3.4 | cross-arch α-safety sweep (Gemma / Qwen3 / GLM-4) | local archive | α=0 bit-equal across 12 cells |
| R-5.1 / v3.4 | Q3 adversarial chat × LOPI on Gemma-4-E2B | local archive | LOPI is the only configuration that elevates the easiest-fact pair to partial implant at α∈{8,10} |
| R-6 / v3.4 | persistent AttnNativeBank (safetensors + filelock) | `tests/test_bank_persistence.py` | round-trip bit-equal under same dtype |
| **S / v3.5** | U-LOPI auto-calibration profiler (`ulopi_v35`) | `deltamemory/memory/lopi_profiler.py`, `tests/test_lopi_profiler.py`, `tests/test_lopi_universal.py` | replaces hard-coded `norm_base=10.0`; same LOPI across Gemma / Qwen3 / GLM-4 / Llama / GPT-2 |
| **R-7 / v3.6** | bank-side V-scale calibration (`ulopi_v36`) | `deltamemory/memory/attn_native_bank.py`, `tests/test_value_scale_calibration.py` | no-v_norm families cap M_V RMS without amplifying small V; Gemma native v_norm stays untouched |

The long-form narrative log lives in [`docs/HISTORY.md`](docs/HISTORY.md).
Per-stage code/config diffs live in [`CHANGELOG.md`](CHANGELOG.md). Raw
experiment dumps, reports, generated paper assets, and transcripts are kept as
local archives and are intentionally not tracked in the production tree.

## Reproducing experiments

Phase-R+ benchmark drivers used by the cleanroom reports:

```bash
python scripts/run_v31_benchmark.py --help        # v3.1 baseline benchmark
```

The v3.1 intervention demo (true / counter-prior facts, per-arch α defaults):

```bash
python scripts/run_intervention_demo.py \
  --model google/gemma-4-E2B \
  --device cuda --dtype bfloat16 \
  --false-facts
```

On Apple Silicon, follow [`docs/apple_silicon.md`](docs/apple_silicon.md) for
the stable MPS stack. Raw experiment outputs are local-only archives; promote
only audited, reproducible summaries into public documentation.

## Tests

```bash
pytest tests/ --ignore=tests/conservation_real_models.py
```

Expected: all local tests pass. The fully skipped/opt-in conservation suite
(`conservation_real_models.py`) downloads multi-GB HF checkpoints — see its
module docstring before running it.
Phase-S coverage in particular: `test_lopi_profiler.py` (profile bit-equality)
and `test_lopi_universal.py` (cross-arch shape + bit-equality on Gemma /
Qwen3 / GLM-4 / Llama / GPT-2).

## Repository map

| Path | Purpose |
|---|---|
| `deltamemory/memory/attn_native_bank.py` | AttnNativeBank + per-layer patcher |
| `deltamemory/memory/lopi.py` | Dynamic LOPI v3.4 injector |
| `deltamemory/memory/lopi_profiler.py` | U-LOPI Phase S residual profiler |
| `deltamemory/memory/bank_persistence.py` | safetensors + filelock bank storage |
| `deltamemory/memory/arch_adapter.py` | per-architecture adapters + α defaults |
| `deltamemory/__init__.py` | top-level public API (Phase S) |
| `scripts/run_intervention_demo.py` | cross-architecture true/false-fact demo |
| `scripts/run_v31_benchmark*.py` | Phase R+ benchmark drivers |
| `docs/HISTORY.md` | long-form per-stage narrative log |
| `tests/` | unit and real-model conservation checks |

## Production deployment / API reference / Migration / Versioning

- API reference: [`docs/api/`](docs/api/) (regenerate with `scripts/build_docs.sh`).
- FastAPI serving scaffold: [`examples/fastapi_serve/`](examples/fastapi_serve/).
- vLLM integration design draft: [`examples/vllm_integration/README.md`](examples/vllm_integration/README.md).
- v0.3 → v0.4 migration: [`docs/migration_v0.3_to_v0.4.md`](docs/migration_v0.3_to_v0.4.md).
- Versioning policy: [`docs/versioning.md`](docs/versioning.md).

## License

MIT. See [`LICENSE`](LICENSE).
