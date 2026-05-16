# Phase C — v3 Memory Architecture (post-sign-off mandate)

**Mandate (2026-05-16, sign-off context)**: continue advancing; prove the path works; don't forget the original intent.

**Original intent**: *delta memory → LLM memory jump*. v2 disproved that the rank-64 K-projector qualifies — its lift comes from a learned template-conditional distribution shift, not from bank content. Phase C designs and tests architectures in which **bank content is load-bearing**, and uses the e16-forgetting A/B symmetry test as the decisive falsifier-of-the-falsifier.

---

## 1. North-star metric

A configuration counts as **real delta memory** iff:

```
Δ_A_initial   ≥ 3.0 nat   AND
Δ_A_after_evict − Δ_B   ≥ 1.0 nat   AND
Δ_B           ≤ 1.0 nat
```

across ≥ 3 seeds. This is the inverse of e16's A/B-symmetry result: eviction must hurt, and a random replacement bank must NOT produce the same lift.

## 2. Roadmap

### e20 — Frozen-projector diagnostic (small, decisive)

Goal: localize where the e16 A/B symmetry comes from.

- e20a: Freeze projector at random init. Train only `bank_gate_heads`. Run e16-forgetting.
  - **Predict**: If Δ_A_initial ≈ 0, projector is necessary → bank-content path is dead. If Δ_A_initial > 1.0 AND Δ_A_after_evict < Δ_A_initial − 1.0, asymmetry could exist via gating alone — promising.
- e20b: Freeze gate heads. Make b-vectors of set_A trainable. Run training, then eviction.
  - **Predict**: If trained-b set_A retains lift ≫ random-b set_B, content lives in b-vectors → the path forward is "make banks trainable, projector light/frozen".

### e21 — Hard-attention bank read (architectural)

If e20 indicates content can be loaded into b-vectors, replace the K-projector's soft averaging over bank with:

```
i* = argmax_i cos(q, K_proj @ b_i)
read = b_{i*}    # hard top-1, straight-through gradient
```

This forces the model to commit to one bank entry per query → wrong bank ⇒ wrong read ⇒ low Δ_B.

### e22 — e16-forgetting on the e21 architecture (proof)

Run the **same e16 phase B forgetting protocol** on the e21 architecture. Pass iff north-star metric holds at ≥ 3 seeds.

## 3. Out-of-scope (for now)

- Llama / Mistral cross-model on v3 (defer until v3 itself passes on Qwen3-4B)
- lm-eval full sweep (only after v3 passes the north-star)
- Interrupt-API demo (was deferred at v2; re-evaluate only post-v3-PASS)

## 4. Termination conditions for Phase C

Phase C **succeeds** when:
- North-star metric holds at ≥ 3 seeds, AND
- Cross-validated on a second model (Qwen3-1.7B with dim-projected b-vectors), AND
- Capability drift ≤ 5% on WikiText-2 (carry over the e03 guard rail).

Phase C **terminates negative** if:
- After e20a, e20b, e21 all individually fail to produce Δ_A_after_evict − Δ_B ≥ 1.0 nat on at least one seed, AND
- No new mechanism is identified within 5 more pilot variants.

In the negative termination case, the conclusion is: on Qwen3-4B-bf16 with the LPL hook architecture, delta-memory-as-item-specific-recall is **structurally unachievable**, and the program closes.

## 5. Working surface

- Driver fork: `v2/experiments/e20_frozen_projector/run.py` (forks e16 phase B).
- Architecture changes for e21: `v2/core/projector.py` (add hard-top1 read mode).
- Results land in `v2/experiments/e2{0,1,2}_*/` with the same JSON schema.
- Each result must produce a standalone `v2/verdicts/E2{0,1,2}_VERDICT.md` following the canonical template.
