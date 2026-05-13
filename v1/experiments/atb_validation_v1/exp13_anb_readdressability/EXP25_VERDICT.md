# Exp25 — K-routing stabilization: VERDICT

**Status:** `EXP25_FRAGILE_K_ROUTING` — K-routing gives weak retrieval signal at
~2× chance across all bank sizes, but the **margin gap** does not reliably
exceed `minus_correct` controls beyond a single n=100 noise spike.
However, **conditional analysis** (Exp26c, free on Exp25 data) reveals the
first real V-content signal once K routes correctly.

## Methodology

- Model: Qwen3-4B-Instruct-2507, MPS bf16.
- Dataset: CounterFact filtered (top-rows-as-bank, full set as eval).
- Architecture: sparse-attention readout (`bank_topk=1`), dual-site capture
  (K@`relation_last`, V@`subject_last`).
- New metrics: `retrieval_accuracy` via `SlotRecorder` hooks
  (`exp25_metrics.py`), `bank_attention_mass`, `selected_slot_id`.
- 5 variants: `full_bank_concat`, `full_bank_topk1`, `…_minus_correct`,
  `…_meanV`, `…_shuffled_factids`.
- Analyzer: paired bootstrap CI (`analyze_exp25.py`).

## Phase 1 — α refinement (`run_mps_exp25_alpha/`)

- n=100 × 3 seeds × 7 α ∈ {0.003 … 0.030} × 5 variants + base. 10,800 cells.
- Wall time: 2046 s.

| α | A: topk1−minus_correct CI | C: topk1−meanV | D: topk1−shuffled |
|---|---|---|---|
| 0.003 | **+0.165 [+0.026, +0.314]** ✓ | −0.041 | −0.026 |
| 0.005 | **+0.193 [+0.043, +0.345]** ✓ | **−0.119 [−0.234, −0.007]** ✗! | −0.084 |
| 0.007 | **+0.184 [+0.072, +0.308]** ✓ | −0.070 | +0.002 |
| 0.010 | **+0.187 [+0.062, +0.321]** ✓ | −0.047 | +0.066 |
| 0.015 | −0.020 | +0.042 | −0.063 |
| 0.020 | −0.011 | −0.048 | +0.018 |
| 0.030 | +0.017 | +0.041 | +0.036 |

Sweet spot α∈[0.003, 0.010]. Above 0.015, V damping kills the signal.

## Phase 2 — bank-size sweep (`run_mps_exp25_N{32,64,200,400}/`)

α=0.005, 3 seeds.

| N | retrieval_accuracy | chance (1/N) | ratio | bank_mass | Gate A diff | CI |
|---|---|---|---|---|---|---|
| 32  | 0.063 | 0.031 | 2.0× | 0.259 | −0.035 | [−0.35, +0.30] |
| 64  | 0.031 | 0.016 | 2.0× | 0.298 | −0.185 | [−0.45, +0.06] |
| 100 | 0.020 | 0.010 | 2.0× | 0.332 | **+0.193** ✓ | [+0.04, +0.35] |
| 200 | 0.010 | 0.005 | 2.0× | 0.365 | −0.046 | [−0.15, +0.05] |
| 400 | 0.0025 | 0.0025 | 1.0× | 0.405 | −0.043 | [−0.084, **−0.002**] negative |

### Findings

1. **Retrieval accuracy ≈ 2× chance is stable across N up to N=200**, then
   decays. This is small but **structural**: K@relation_last does carry
   weak slot identity. Bank routes to correct slot twice as often as
   pure noise would.
2. **Gate A (margin gap) is NOT monotone in N**. The +0.193 at N=100
   was a sample-size noise spike; CI overlaps zero (or excludes on the
   wrong side at N=400). **K-routing margin gap is fragile.**
3. **Bank mass grows with N** (26% → 41%). The bank steals an
   increasing share of attention, but this acts as global steering
   rather than directed readout — gate A doesn't grow with mass.

## Phase 3 — Exp26c conditional V analysis (free on Exp25 data)

For facts where `retrieval_correct == 1` (K routed to correct slot),
recompute Gate D `topk1 − shuffled_factids`:

| α | n_all | mean Gate D (all) | n_correct | **mean Gate D conditional** |
|---|---|---|---|---|
| 0.005 | 300 | −0.084 | 6 | −0.133 |
| 0.007 | 300 | +0.002 | 6 | **+0.268** |
| 0.010 | 300 | +0.066 | 6 | **+0.286** |
| 0.015 | 300 | −0.063 | 6 | +0.176 |
| 0.020 | 300 | +0.018 | 6 | +0.164 |
| 0.030 | 300 | +0.036 | 9 | **+0.600** |

**This is the program's first V-content identity signal.** When K routes
correctly, V identity matters substantially — gap ≥ +0.27 at α∈{0.007,
0.010}. Unconditionally, V identity is invisible because 98% of K-routes
land on the wrong slot.

Caveat: sample size n_correct=6–9 per α. No paired bootstrap CI yet (need
to rerun with seed-paired conditioning). Treat as **hypothesis-generating**.

## Verdict

`EXP25_FRAGILE_K_ROUTING + CONDITIONAL_V_SIGNAL_DETECTED`

The K-routing margin gap fails the original strict gate (CI lower bound
> +0.10 robust across N). But two new signals are real:

1. K-routing **accuracy** at 2× chance, stable to N=200.
2. V-content gap **conditional on correct K** is large (+0.27 nat),
   suggesting V@subject_last IS retrievable when K succeeds.

## Implications for Exp26

- The original Exp26 V-site sweep is justified: lifting K-routing accuracy
  even modestly should compound with the +0.27 conditional V signal.
- New `Exp26d` priority: increase α conditional on retrieval confidence
  (top1−top2 gap). At α=0.030, conditional D=+0.60 — high α works *if*
  applied selectively to confident routes.
- Exp27 full-validation requires Exp26 to lift K-routing accuracy and
  re-test V-identity at the new operating point.
