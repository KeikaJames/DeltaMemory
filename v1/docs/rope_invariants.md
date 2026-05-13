# RoPE Relative-Position Invariants

This document pins the RoPE-related design contract that the v0.4 injector
stack relies on for *position-invariant memory recall*.  It accompanies the
regression suite in `tests/test_rope_relpos_invariance.py`.

## TL;DR

* `AttnNativeBank` scores bank entries **with both Q and K pre-RoPE**.
* The pre-RoPE inner product `q_pre · k_pre` is independent of the token
  position at which Q is queried *and* the position at which K was written.
* This is the only design choice in the v0.4 stack that matters for RoPE
  relative-position correctness; SCAR, CAA, LOPI's V-rotation, and the
  legacy `MnemeWriter` are all position-agnostic by construction.
* If a future refactor accidentally swaps the bank-scoring LHS from `q_pre`
  to `q_post`, recall@k silently degrades for any fact whose write-position
  differs from its query position.  Two textual tripwires in
  `tests/test_rope_relpos_invariance.py` guard against that drift.

## The contract

```text
deltamemory/memory/attn_native_bank.py:18-24

    The bank stores **pre-RoPE post-norm K**.  At attention time we keep
    a **pre-RoPE Q copy** alongside the post-RoPE Q for bank scoring:

        scores_orig = q_post @ k_post^T * scaling     # standard, with RoPE
        scores_bank = q_pre  @ M_K^T    * scaling     # both sides pre-RoPE
```

Because RoPE is applied as an orthogonal rotation that commutes between
the two operands of an inner product *only when the rotation angles
match*, the pre-RoPE inner product is exactly position-invariant:

```text
q_pre · k_pre  ==  R(p_q) q_pre · R(p_w) k_pre   ⟺   p_q == p_w
```

Doing the bank inner product post-RoPE would mean a fact written at token
position `p_w = 5` only scores correctly for queries at `p_q ≡ 5 (mod
wavelength)` — a recall failure that would be silent on tests where every
fact is written and queried at the same position.

## Per-injector position-handling table

| Injector | Site of injection | RoPE relevance | Position handling |
|---|---|---|---|
| `AttnNativeBank` (`attn_native_bank.py`) | inside attention, on Q/K/V | **YES** | Q pre-RoPE captured at line 403, K pre-RoPE captured at line 432, bank scored with `q_pre @ mk_e^T` at line 516 |
| KV-shared layers in `AttnNativeBank` | shared K/V slot | n/a (no capture) | `k_pre_for_capture = None` at line 430 — no K is captured on shared layers, so no post-RoPE leak is possible |
| GQA in `AttnNativeBank` | `repeat_kv` on bank K | unchanged | `repeat_kv` is a pure tensor expansion along the kv-group dimension; it commutes with the pre-RoPE invariant (test `test_gqa_repeat_kv_preserves_pre_rope_invariance`) |
| Partial-RoPE (GLM-4) | first half of `head_dim` | preserved | `arch_adapter.Glm4Adapter` uses HF's `apply_rotary_pos_emb` only on `q_post`/`k_post`; `q_pre`/`k_pre` are untouched and full-dim — the pre-RoPE invariant covers the whole head_dim (test `test_partial_rope_glm4_pre_rope_invariance`) |
| `SCARInjector` (`scar_injector.py:215`) | `o_proj` output (residual-stream-like) | none | injection is `α * (target - h) @ B B^T` on the post-attention output; positional rotation has already been integrated out |
| `CAAInjector` (`caa_injector.py`) | residual stream at chosen layer | none | activation steering on the residual stream; position-agnostic |
| `lopi_inject.V_rot` | Bank V output, `cos(θ) V + sin(θ) M_⊥` | none (semantic, not positional) | this is a *semantic* rotation in the V-readout subspace, not a positional one; θ is derived from γ_t, not from token position |
| LOPI `derivative_gate` (`lopi.py:256`) | `‖Q_t − Q_{t-1}‖` | preserved | norm of a difference is invariant under any orthogonal transform; the derivative gate is rotation-equivariant whether Q is pre- or post-RoPE, as long as both Q_t and Q_prev share the convention |
| `solve_rome_writer` (`rome_writer.py`) | bank V via ridge regression on K | docstring contract | the caller is contractually required to pass pre-RoPE K (`rome_writer.py:11`); there is no runtime check, but `AttnNativeBank.M_K` is always pre-RoPE so the only legitimate caller path is correct |
| `MnemeWriter` (`writer.py`) | residual stream → `nn.Linear` | none | reads `h_in`/`h_out` from outside attention; no Q/K touch |

## Tripwires

Two tests in `tests/test_rope_relpos_invariance.py` are textual guards
intended to fire on any future refactor that breaks the contract:

1. `test_attn_native_bank_uses_q_pre_for_bank_scoring` —
   asserts that the patched forward of `_make_patched_forward` contains
   `scores_bank = torch.matmul(q_pre, …)` (or the cosine-mode equivalent
   `q_cos = q_pre / ‖q_pre‖`) and **never** `scores_bank = torch.matmul(q_post, …)`.

2. `test_attn_native_bank_captures_k_pre_rope` —
   asserts that the non-shared branch sets `k_pre_for_capture = k_pre`,
   the shared branch sets `k_pre_for_capture = None`, and the capture
   write `ctx._capture_K[layer_idx] = k_pre_for_capture[…]` references
   `k_pre_for_capture`, not `key_states` or `k_post_`.

If either tripwire fires, do **not** rewrite the test to make it pass —
the test is the contract.  Restore the pre-RoPE invariant in the patcher
and re-run.

## Adding a new architecture

When adding an `ArchAdapter` subclass in `deltamemory/memory/arch_adapter.py`:

1. `apply_rope` must take `(q_pre, k_pre, cos, sin) → (q_post, k_post)`
   and never mutate `q_pre`/`k_pre` in place.  All current adapters
   satisfy this — they call HF's per-family `apply_rotary_pos_emb` which
   returns new tensors.
2. Partial-RoPE adapters (GLM-4 style) must rotate only the configured
   prefix of `head_dim`; the un-rotated suffix flows through unchanged
   on both Q and K.  The pre-RoPE invariant on the *full* head_dim
   continues to hold.
3. Do not add any code path that consumes the bank with post-RoPE Q.
   The bank stores pre-RoPE K and is meaningless under any other
   convention.
4. Add the new architecture to the parametrize matrix of
   `test_pre_rope_dot_is_position_invariant` if its rotary
   conventions differ materially (rope_theta, head_dim parity, partial
   rotation).

## Open follow-ups

* `rome_writer.solve_rome_writer` could surface a `dtype` check or a
  documented sentinel asserting the caller is feeding pre-RoPE K.  The
  current callers (only `AttnNativeBank.M_K`) are correct, so this is a
  defensive nicety, not a bug.
* `lopi.py:32` docstring describes "Q_t, Q_prev — pre/post-RoPE Q
  (configurable)" but there is no actual configuration switch; the
  derivative gate is correct under either convention by rotation
  equivariance, but the documentation should be tightened.
