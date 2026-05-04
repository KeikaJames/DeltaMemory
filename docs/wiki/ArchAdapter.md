# ArchAdapter — Adding a New Model Family

`ArchAdapter` is the seam where Mneme meets a specific transformer family. The patched forward is generic; per-family quirks (q/k/v norm presence, RoPE shape, KV-sharing, GQA repeat) are routed through the adapter.

To add a new family you write **one** subclass and register it. No changes to the patcher or the bank.

## Interface

`deltamemory/memory/arch_adapter.py`

```python
@dataclass
class ArchAdapter:
    name: str = "base"

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        """Return True for the attn class your family uses."""
        return False

    # Norms — defaults read attn.q_norm / .k_norm / .v_norm if present.
    def apply_q_norm(self, attn, q): ...
    def apply_k_norm(self, attn, k): ...
    def apply_v_norm(self, attn, v): ...

    # RoPE — must override.
    def apply_rope(self, q, k, cos, sin) -> tuple[Tensor, Tensor]: ...

    # KV-sharing (Gemma-4 has shared layers; most don't).
    def is_kv_shared(self, attn) -> bool: ...
    def kv_shared_index(self, attn) -> int | None: ...
    def store_full_length_kv(self, attn) -> bool: ...

    # GQA repeat — default uses transformers-canonical view+expand.
    def repeat_kv(self, x, n_rep) -> Tensor: ...
```

The patched forward holds Q/K in `(B, T, H, D)` layout. `apply_rope` must accept and return tensors in that layout.

## The 4 shipped adapters

| family | norms | KV-sharing | RoPE source | RoPE caller-shape |
|---|---|---|---|---|
| `Gemma4Adapter` | q+k+v | yes | `transformers.models.gemma4.modeling_gemma4` | `unsqueeze_dim=2`, `(B, T, H, D)` |
| `Qwen3Adapter` | q+k only | no | `transformers.models.qwen3.modeling_qwen3` | `unsqueeze_dim=1`, `(B, H, T, D)` (adapter transposes) |
| `LlamaAdapter` | none | no | `transformers.models.llama.modeling_llama` | same as Qwen3 |
| `Glm4Adapter` | none | no | `transformers.models.glm4.modeling_glm4` | same as Qwen3 |

`LlamaAdapter` covers `LlamaAttention`, `Qwen2Attention`, `MistralAttention` (matched by class-name substring). DeepSeek-R1-Distill-Qwen-32B is Qwen2-arch under the hood and routes here.

> **Note on GLM-4**: `Glm4Adapter` matches HuggingFace's `Glm4Attention` (the modern HF-native GLM-4 like `THUDM/GLM-4-9B-0414`). The older `THUDM/glm-4-9b-chat` ships with `modeling_chatglm.py` (custom `trust_remote_code` ChatGLM arch) and is **not** covered by `Glm4Adapter`. To support it you would write a `ChatGLMAdapter`.

## Adding a new family in 4 steps

Suppose you want to add **Phi-3**.

### 1. Subclass

```python
# deltamemory/memory/arch_adapter.py

class Phi3Adapter(ArchAdapter):
    def __init__(self):
        super().__init__(name="phi3")

    @classmethod
    def matches(cls, attn_module):
        return "Phi3" in type(attn_module).__name__

    # Phi-3 has no q/k/v norms — defaults (identity) are correct.

    def apply_rope(self, q, k, cos, sin):
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb
        # Phi-3 uses unsqueeze_dim=1 with (B,H,T,D); transpose-roundtrip.
        q_t = q.transpose(1, 2); k_t = k.transpose(1, 2)
        q2, k2 = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)
        return q2.transpose(1, 2), k2.transpose(1, 2)
```

### 2. Register

```python
_REGISTRY: list[type[ArchAdapter]] = [
    Gemma4Adapter,
    Qwen3Adapter,
    Phi3Adapter,         # <-- here
    LlamaAdapter,
    Glm4Adapter,
]
```

Or at runtime:

```python
from deltamemory.memory.arch_adapter import register_adapter, ArchAdapter

@register_adapter
class Phi3Adapter(ArchAdapter): ...
```

### 3. Run the conservation test

```bash
python tests/conservation_real_models.py --models phi-3 --device cuda
```

Add `"phi-3": "microsoft/Phi-3-mini-4k-instruct"` to `MODEL_REGISTRY` in that file.

The pass criterion is `bit_equal = True` (max-abs-diff = 0.000) when the bank is empty / α = 0. If you fail this, the most common causes are:

1. **RoPE layout mismatch** — your apply_rope received `(B,T,H,D)` but called the upstream helper with `unsqueeze_dim=1` (which expects `(B,H,T,D)`). Add the transpose round-trip.
2. **Missed a norm** — your family has `q_norm` but the default fallback didn't pick it up because the attribute is named differently (e.g. `q_layernorm`). Override `apply_q_norm`.
3. **GQA repeat differs** — some families call `torch.repeat_interleave` with a different axis; override `repeat_kv` if the canonical `view + expand` isn't bit-equal.

### 4. (Optional) train a K-projector for that family

The v3 frozen K-projector was trained on Gemma-4-E2B. It does **not** transfer to Qwen3 / Phi-3 / Llama — see `transcripts/v3_intervention/qwen3-4b-instruct-2507/demo.md` for what happens when you naively re-use it (~−12 logprob collapse).

Phase L of v3.1 trains the K-projector with cross-relation **and cross-architecture** hard negatives so a single projector generalizes. Until that lands, each new family gets identity-init by default — safe (bit-equal at α=0) but no signal lift.

## What the adapter does NOT do

- **Touch LLM weights** — the adapter only **reads** from `attn_module` (norms, layer index, scaling). It never writes.
- **Change the attention math** — `apply_rope` and `repeat_kv` produce tensors that are then fed to the same `softmax(QKᵀ/√d)V` the upstream uses. We add bank K/V via `concat` along the key-time dimension.
- **Decide capture policy / writer behavior** — those are orthogonal and live in `capture_policy.py` and the bank.

## Reference

- `deltamemory/memory/arch_adapter.py` — full source
- `tests/conservation_real_models.py` — the gate
- `deltamemory/memory/attn_native_bank.py:218` — `_make_patched_forward` (where the adapter is consulted)
