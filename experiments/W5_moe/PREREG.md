# W.5 — MoE per-expert column-cap shield · Pre-Registration

**Status**: PREREG (frozen before any cells.jsonl row is written)
**Hardware target**: 128 GB unified-memory Apple Silicon (or equivalent CUDA);
  does NOT run on the 64 GB development machine due to MoE checkpoint size
**Predecessor modules**:

- `deltamemory/arch/moe_adapter.py` (Phase X.2; Qwen3MoE + Mixtral adapters)
- `experiments/W3_decision/DECISION.md` (mHC shield demoted but kept as ablation flag;
  for MoE the per-expert cap is the *only* spectral safeguard, so it remains studied)

---

## 1. Methodology paradigm

**Controlled variable** + **Benchmarking** (per `plan.md` W.5 paradigm cell).
The *single* axis is the cap mechanism (none / global / per-expert); model and
dataset are held fixed within each sub-grid.

---

## 2. Model selection

Primary: `Qwen/Qwen3-MoE-A3B` (30 B total, 3 B active, ~18 GB int4).
Fallback (in order): `mistralai/Mixtral-8x7B-Instruct-v0.1` (~24 GB int4),
`deepseek-ai/DeepSeek-V2-Lite-MoE` (~16 GB int4).
The first model that loads within 60 s and passes the α=0 bit-equality witness
becomes the locked W.5 model. Subsequent reruns must use the same model.

> *Substitution policy*: substitution is allowed before the first non-α=0 row
> is written; after that, only an explicit corrigendum + new branch may change
> the model.

---

## 3. Grid

| Axis | Levels | Count |
|---|---|---|
| Model | 1 (locked at run start) | 1 |
| α | 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 | 7 |
| Cap mechanism | `none`, `global` (W.1 column cap, single bank-wide), `per_expert` (X.2 per-expert cap) | 3 |
| κ (cap value) | 0.5, 1.0, 2.0 | 3 |
| Seed | 0, 1, 2 | 3 |
| Prompts | gold_30prompts.jsonl (sha-locked) | 30 |

Total: 1 × 7 × 3 × 3 × 3 × 30 = **1890 forward passes**.

---

## 4. Fixed configuration

- V-scale: ON, RMS cap 0.5 (Qwen-family has no v_norm)
- LOPI: OFF (W.3 demoted; W.5 isolates the cap mechanism)
- CAA: OFF (W.5 isolates the cap mechanism)
- ECOR: OFF
- Bank: random_v init, n_slots = 8, dtype = bfloat16, single neutral write-fact
- Eval metric: drift = mean per-token NLL on gold_30prompts.jsonl held-out span

### Per-expert cap mechanics (X.2 contract)

For each expert `e ∈ {0..E-1}` at decoder layer ℓ:
1. Read router gate `g_e(token)` from FFN router output (acknowledged FFN-router
   timing approximation; see `docs/theory/mhc_moe.md` and X.2 follow-up note)
2. Bucket bank columns by their effective routed weight:
   `W^{(e)}[:, T+j] = g_e(token) · W[:, T+j]`
3. Independent column cap to κ on each bucket
4. Recompose attention output

### Global-cap mechanics (W.1 contract)

Single column cap to κ over the *combined* bank columns regardless of expert
routing — the W.1 baseline carried into W.5 for direct comparison.

---

## 5. Pre-registered hypotheses

H5a: per-expert cap reduces drift relative to global cap on **at least 4/7 α
levels** at the optimal κ (paired Wilcoxon, p < 0.05 corrected for the 7-α family).

H5b: at α = 0, every cell's `|drift_inj - drift_no_memory|` is **strictly less
than 1e-4** for both cap mechanisms (red-line; failure aborts the cell).

H5c: per-expert cap shows a **monotonic** drift response in κ (lower κ → lower
drift) on the locked model; non-monotonic behavior triggers the X.2 follow-up
investigation (FFN-router-gate-as-attn-cap-proxy ordering).

---

## 6. Verdict matrix

| Outcome | Decision |
|---|---|
| H5a + H5b + H5c PASS | per-expert cap shipped as MoE default; W-T2 opens for second-MoE generalization |
| H5a PASS, H5c FAIL | per-expert cap shipped, but X.2 ordering issue takes priority for W-T2 |
| H5a FAIL | per-expert cap demoted to ablation; global cap remains; W.5 REPORT documents what was learned |
| H5b FAIL | run aborted, X.2 module patched, run restarted on new branch |

---

## 7. Anti-tampering

1. PREREG.md committed before cells.jsonl is opened
2. cells.jsonl appended one row per cell with `cell_id = sha1(model|cap|kappa|alpha|seed|prompt_id)`
3. Aggregate computation lives in `aggregate.py`; verdict tables re-derivable from cells
4. env.json captures torch / transformers / commit / dtype / device / model SHA / locked model name

---

## 8. Out-of-scope (deferred)

- MLA / KV-shared MoE architectures (DeepSeek-V3 latent-KV) → W-T2.4
- shared-expert (always-active) bank routing → W-T2.2
- dropped-token (capacity overflow) bank routing → W-T2.2
- MoE × LOPI / MoE × CAA combination → W-T2 round 4

---

## 9. Sign-off

PREREG closed at commit time. Next commit on this branch must be cells.jsonl
output or a corrigendum amending §2/§3 with explicit reason.
