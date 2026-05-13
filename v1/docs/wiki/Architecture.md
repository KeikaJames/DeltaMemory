# Architecture

Mneme has three layers: **Patcher** (touches the LLM), **Bank** (stores K/V), **Writer/Reader** (puts facts in / reads them out). Adapters isolate per-family attention conventions.

```
                       ┌─────────────────────────────────────────────────┐
                       │  Frozen base LLM (Gemma-4 / Qwen3 / Llama / …)  │
                       │                                                 │
   prompt ──────►      │  ┌──────────┐         ┌──────────┐              │
                       │  │  attn 0  │◄──┐  ┌──┤  attn N  │              │
                       │  └──────────┘   │  │  └──────────┘              │
                       └────────┬────────┴──┴────────────────────────────┘
                                │  patched forward
                                │  (per-layer)
                                ▼
                ┌─────────────────────────────────────┐
                │   AttnNativePatcher  +  ArchAdapter │
                │                                     │
                │   q,k,v_norm                        │
                │   apply_rope(q, k, cos, sin)        │
                │   repeat_kv(x, n_rep)               │
                │   is_kv_shared(layer)               │
                └────────────┬────────────────────────┘
                             │  uses
                             ▼
                ┌─────────────────────────────────────┐
                │   AttnNativeBank                    │
                │   M_K[layer]  ∈  [N, Hkv, head_dim] │
                │   M_V[layer]  ∈  [N, Hkv, head_dim] │
                │   k_projector (optional, trained)   │
                │   bank_temperature (default 1.0)    │
                │   bank_topk (default 0 = bit-equal) │
                └─────────────────────────────────────┘
```

## Patcher: `deltamemory.memory.attn_native_bank.AttnNativePatcher`

Monkey-patches every attention module's `forward` while the context manager is active. The patched forward is generic; family differences are routed through the `ArchAdapter`.

### What the patched forward does

```python
# 1. project + norm Q (and K, V if not KV-shared)
q_pre = adapter.apply_q_norm(self, self.q_proj(h).view(B, T, Hq, D))
k_pre = adapter.apply_k_norm(self, self.k_proj(h).view(B, T, Hkv, D))
v     = adapter.apply_v_norm(self, self.v_proj(h).view(B, T, Hkv, D))

# 2. RoPE (per-family)
q_post, k_post = adapter.apply_rope(q_pre, k_pre, cos, sin)

# 3. KV-cache + KV-sharing (Gemma-4 has shared layers; others usually not)
key, value = past_key_values.update(k_post, v, layer_idx)

# 4. standard attention scores
k_rep = adapter.repeat_kv(key, num_kv_groups)        # GQA broadcast
v_rep = adapter.repeat_kv(value, num_kv_groups)
scores_orig = q_post @ k_rep.T * scaling + attention_mask

# 5. bank attention — per-layer concat into softmax
if bank_attached and α > 0 and bank.M_K[layer].size(0) > 0:
    mk = k_projector(layer, bank.M_K[layer])         # identity-init when v3 not loaded
    mk = adapter.repeat_kv(mk.unsqueeze(0).T, n_rep)
    mv = adapter.repeat_kv(bank.M_V[layer].unsqueeze(0).T, n_rep)
    scores_bank = q_pre @ mk.T * scaling
    if bank.bank_topk > 0:                           # Stage 15A structural fix
        scores_bank = topk_mask_(-inf)(scores_bank, bank.bank_topk)
    scores = cat([scores_orig, scores_bank], dim=-1)
    weights = softmax(scores)
    out = weights[:T_orig] @ v_rep + α * weights[T_orig:] @ mv
else:
    weights = softmax(scores_orig)
    out = weights @ v_rep                             # bit-equal fallback
```

### The conservation law

When `bank.empty` or `α == 0`, the `else` branch executes verbatim and the patched forward is element-wise identical to the unpatched forward. This is the **red line** of the project — any change that breaks bit-equality without explicit user opt-in is a bug.

Verified by `tests/conservation_real_models.py`:

| model | adapter | device | max-abs-diff | bit_equal |
|---|---|---|---:|---|
| google/gemma-4-E2B | gemma4 | mps bf16 | 0.000e+00 | True |
| google/gemma-4-E2B | gemma4 | cuda bf16 | 0.000e+00 | True |
| Qwen/Qwen3-4B-Instruct-2507 | qwen3 | cuda bf16 | 0.000e+00 | True |

## Bank: `AttnNativeBank`

Per-layer ragged tensor of K and V slices, plus optional trained pieces:

```python
@dataclass
class AttnNativeBank:
    M_K: list[Tensor]       # [n_layers], each Tensor [N, Hkv, head_dim]
    M_V: list[Tensor]
    fact_ids: list[str]     # length N (same for all layers)
    addresses: list[str]
    k_projector: Optional[KProjector] = None    # per-layer Linear, identity-init
    bank_temperature: float = 1.0               # >0; 1.0 = no-op
    bank_topk: int = 0                          # 0 = no gating (bit-equal v3)
```

Storage cost per fact (Gemma-4-E2B, 35 layers, 8 heads, head_dim=256, bf16):

```
35 × 8 × 256 × 2 bytes × 2 (K and V) ≈ 280 KB / fact
```

A 10k-fact bank is ~2.7 GB. This is **not** intended to scale to web-scale corpora; for that, use RAG. Mneme targets thousands of facts you want **active in the LLM's attention**, not retrievable.

## Writer: `write_fact`

Single-shot capture. Forward the write prompt with `patcher.capturing(capture_pos=...)` active; the patched forward records `K_pre[layer][:, pos, :, :]` into `ctx._capture_K[layer]`, and the post-RoPE+norm V at the same position into `ctx._capture_V[layer]`. After the forward, those slices are appended to `bank.M_K` / `bank.M_V`.

### Capture policy

* `period` (default, v2/v3 frozen): capture at the period token of the write prompt.
* `address`: capture at the first token of `address` (a short paraphrase passed at write time).
* `multi`: capture at *all* address sites + period; the bank stores multiple slices per fact, each tagged `fact_id@role`.

`period` won the Stage 14 dev sweep on Gemma-4-E2B (recall@1 = 0.4343, vs `multi` = 0.3737, `address` = 0.4040).

## Reader: `forward_with_bank`

```python
with patcher.patched(), patcher.injecting(bank, alpha=1.0):
    out = model(input_ids=..., attention_mask=..., use_cache=False)
```

Returns the last-position logits. `alpha` controls the bank's contribution; `alpha=0` reduces to the unpatched forward.

## What's *not* in the LLM

* **No new modules in the LLM forward graph** — the bank is `concat`'d into existing softmax tensors; no extra `Linear`, no extra activation.
* **No gradients into LLM weights** — `forward_with_bank` runs under `torch.no_grad()` for inference; training paths only flow into `k_projector` (and optionally `Writer`).

## Files

- `deltamemory/memory/attn_native_bank.py` — patcher + bank + write_fact + forward_with_bank
- `deltamemory/memory/arch_adapter.py` — ArchAdapter base + 4 family adapters
- `deltamemory/memory/k_projector.py` — trained per-layer Linear (identity init)
- `deltamemory/memory/capture_policy.py` — `period` / `address` / `multi`
- `deltamemory/memory/writer.py` — extension hook for ROME-style writers (off in v3, deferred to v3.1)
