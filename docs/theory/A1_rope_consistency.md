---
audit_item: A1
verdict: resolved
evidence_path: experiments/A1_rope_audit/raw_cells.json
---

# A1 — RoPE relative-position consistency audit

## Diagnosis

The complaint assumes the bank stores pre-RoPE K and later concatenates it into the model's native K sequence so HuggingFace RoPE rotates the bank slots at the read-time sequence positions. That would indeed corrupt a fact written at time `p_w` and read at time `p_r`: normal attention compares `R(p_r) q` with `R(p_w) k`, while the broken path would compare `R(p_r) q` with `R(p_current_slot) k`.

The code does **not** do that. `attn_native_bank.py` explicitly keeps two streams:

- native sequence: `scores_orig = q_post @ k_repeat.T`, where `q_post,k_post = adapter.apply_rope(q_pre,k_pre,cos,sin)`;
- bank sequence: `scores_bank = q_pre @ mk_e.T`, where `mk_e` is expanded from `bank.M_K` and is never passed to `adapter.apply_rope`.

Capture stores `k_pre_for_capture[:, pos, :, :]` before RoPE. Injection reads `bank.M_K[...]` directly into `scores_bank`. Therefore:

(a) bank K is **not** model-RoPE rotated at read time;  
(b) no read-time `position_id` is assigned to bank K;  
(c) different write-time facts share no absolute bank position id. Their only ordering is the bank slot index used as a softmax column.

## Evidence /反证

Code references:

- `deltamemory/memory/attn_native_bank.py:365-403`: computes `q_pre`, `k_pre`, then applies RoPE only to native `q_post_, k_post_`.
- `deltamemory/memory/attn_native_bank.py:415-431`: captures `k_pre_for_capture` into `_capture_K` before RoPE.
- `deltamemory/memory/attn_native_bank.py:461-479`: expands `bank.M_K` and computes `scores_bank = q_pre @ mk_e.T`.
- `deltamemory/memory/attn_native_bank.py:514-515`: concatenates already-computed score tensors, not raw K tensors.

Minimal probe (`experiments/A1_rope_audit/probe.py`) ran on `Qwen/Qwen2.5-0.5B`, bf16, MPS. It wrote two facts and varied filler gap before the second fact:

| gap tokens | slot0 mean bank col-sum | slot1 mean bank col-sum | interpretation |
|---:|---:|---:|---|
| 0 | 29.50 | 29.77 | near-symmetric |
| 4 | 25.34 | 32.17 | semantic/filler capture changes, not RoPE slot rotation |
| 64 | 25.18 | 31.85 | stable from gap=4 to gap=64 |

The discontinuity is from changing the write prompt's hidden state before capture; there is no monotonic phase drift with absolute position. Raw per-layer cells are in the evidence path.

## Math: why the broken design would fail

For RoPE rotation matrix `R(p)`, the desired relative-position score is:

`<R(p_q) q, R(p_k) k> = <q, R(p_k - p_q) k>`.

If a pre-RoPE bank key `k` written at `p_w` is later injected as if it occupied read-time slot `p_b`, the score becomes:

`<R(p_q) q, R(p_b) k> = <q, R(p_b - p_q) k>`.

This equals the desired score iff `p_b = p_w` modulo every RoPE frequency period. Across arbitrary write/read gaps that is false, so the complaint is mathematically valid for raw-K concatenation.

The implemented design instead uses:

`score_bank = <q_pre, k_pre>`.

This removes `R(·)` from both sides, making the bank a position-free semantic pool. It is not relative-position preserving; it is position-agnostic by construction.

## 修复方案 sketches and α=0 compatibility

No urgent code fix is needed for the current implementation. If future work wants position-aware memory, use an explicit opt-in channel:

### (i) Store bank K already RoPE-rotated at write time

```python
# capture path
q_post, k_post = adapter.apply_rope(q_pre, k_pre, cos, sin)
bank.M_K_rot[layer].append(k_post[:, pos])
bank.M_K_pos[layer].append(position_ids[:, pos])

# read path
scores_bank = torch.matmul(q_post, repeat_kv(M_K_rot).transpose(2, 3)) * scaling
```

Compatibility: keep existing `M_K` path as default. At `alpha=0`, `do_inject` is false, so no output change. No new `nn.Parameter`; W_q/W_k/W_v/W_o untouched.

### (ii) Anchored relative position 0

```python
anchor = torch.zeros_like(position_ids_for_bank)
cos0, sin0 = rope(anchor)
_, mk_anchor = adapter.apply_rope(q_dummy, mk_pre, cos0, sin0)
scores_bank = q_post @ mk_anchor.T
```

Compatibility: opt-in only. At `alpha=0`, skipped. It changes semantics from position-free to fixed-anchor and should be a named mode, not a silent replacement.

### (iii) Explicit NoPE bank channel

```python
scores_orig = q_post @ k_post.T * scaling
scores_bank = q_pre @ M_K_pre.T * scaling  # current behavior
scores = cat([scores_orig + mask, scores_bank], dim=-1)
```

Compatibility: this is the current path. It preserves α=0 bit-equality because the branch is gated by `alpha > 0.0` and empty-bank checks.

## Recommendation

Keep the NoPE bank channel as the documented v0.4 behavior. Add a future experiment only if position-aware memories become a claim. The audit item is resolved as a false positive against current code, with a genuine warning for any future raw-K concat refactor.
