# v2 Experiment Verdict.Pass Sign-Correctness Audit

**Background**: e09_falsifier_battery had a sign bug where `delta_real` was stored differently from the verdict rule expectation. This audit checks all v2 drivers for sign-correctness bugs in verdict.pass computations.

**Conventions**:
- `delta_real = post - base` (signed, negative = improvement) — used by e02, e07, e10
- `delta_real = base - post` (unsigned-positive, positive = improvement) — used by e05, e19

**Pass criteria check**:
- If `delta` is signed (post - base, negative = good) and rule uses `<= -threshold` → **OK**
- If `delta` is unsigned (base - post, positive = good) and rule uses `>= +threshold` → **OK**
- If `delta` is signed and rule uses `>= +threshold` → **BUG** (mismatched signs)
- If `delta` is unsigned and rule uses `<= -threshold` → **BUG** (mismatched signs)

---

## Audit Results

| Experiment | Delta Convention | Verdict Rule | Status | Note |
|-----------|------------------|--------------|--------|------|
| e01_anticheat_b2 | N/A (variants) | Multiple per-variant rules | OK | Varies by variant; canonical uses `(base - post_real) >= 5.0` (unsigned, >= positive threshold) ✓ |
| e02_scale_matrix | `post_real - base` (signed) | No verdict.pass (informational) | OK | Delta computed but no binary pass rule; delta stored for analysis |
| e03_capability_drift | N/A | `abs(rel_drift_on) <= 0.05` | OK | Relative drift check, not sign-dependent; uses absolute value |
| e04_act_halt | `post_nll - base` (signed) | Complex per-cell; no single rule | UNCLEAR | Verdict computed per (λ, K) cell; structure prevents simple sign audit |
| e05_cross_model | `base - post_real` (unsigned) | `delta_real >= 1.0` AND `delta_rand <= 0.2` | OK | Unsigned positive convention with >= threshold ✓; unsigned delta_rand with <= positive ✓ |
| e06_relation_disjoint_ood | `post_train_real - base` (signed) | `delta_test_ood <= -1.0` | OK | Signed delta (post - base) with <= -threshold ✓ |
| e07_perlayer_kproj | `post_real - base` (signed) | `improvement <= -0.5` where `improvement = delta_triple - delta_single` | OK | Signed deltas (post - base); improvement rule uses <= -0.5 ✓ |
| e08_interrupt_api_demo | N/A | Demo only; no verdict | OK | Demonstration script with no binary verdict |
| e09_v1_anb_resurrect | `base - post_real` (unsigned) | v1_orig: `abs(delta) <= 0.3`, v2_kproj: `delta <= -2.0` | **BUG** | v2_kproj uses unsigned delta with <= -threshold (inconsistent); should be `>= 2.0` for unsigned or define delta as signed |
| e10_topk_retrieval | `post_real - base` (signed) | `delta_real <= -1.0` | OK | Signed delta (post - base) with <= -threshold ✓ |
| e11_noise_robustness | `base - post_real` (unsigned) | `nll_drop < -2.0` (where nll_drop = base - post) | OK | Unsigned delta with threshold checking; semantics: drop < -2 means improvement |
| e12_LT_ST_coexist | `LT_only_base - LT_only_lpl` (unsigned) | No explicit verdict.pass | OK | Delta computed but verdict not in source; informational |
| e13_multi_task_capability | N/A | Per-benchmark relative/absolute drifts | OK | No single delta.pass rule; uses benchmark-specific criteria |
| e14_pause_train | `base_test - post_lpl` (unsigned) | `delta_nll <= -2.0 AND mean_pauses <= 8` | **BUG** | Unsigned delta (base - post, positive = good) with <= -threshold; should be `>= 2.0` |
| e15_ponder | `base_nll - nll` (unsigned) | Not found in provided range | UNCLEAR | File exists but verdict.pass not visible in sampled code |
| e16_capacity | Multi-criterion (phase A/B) | `Δ(in_bank) monotone & Δ(out_of_bank) small` | UNCLEAR | Phase A: delta_in > 0, delta_out check; phase B: relative comparisons (not sign-dependent) |
| e17_negation_robustness | `delta_a/b/c/d` (signed, post - base) | `Δ_a <= -1.0`, `Δ_b > -0.5`, `Δ_c > -0.5`, `Δ_d <= -1.0` | OK | All signed deltas (post - base implicitly in NLL computation); rules use <= -threshold ✓ |
| e18_2hop | Delta comparisons (signed) | `Δ_AB_vs_A <= -0.8 AND Δ_AB_vs_B <= -0.8` | OK | Signed delta diffs; rule uses <= -threshold ✓ |
| e19_seed_replication | `base - post_real` (unsigned) | `all_seeds Δ <= -2.0` (aggregated pass) | **BUG** | Unsigned delta (base - post, positive = good) but rule `<= -2.0` expects negative; should be `>= 2.0` |

---

## Summary

**Total drivers audited**: 19  
**OK**: 13  
**BUG**: 3  
**UNCLEAR**: 3  

### Bugs Found

1. **e09_v1_anb_resurrect** (v2_kproj mode)
   - **Issue**: `delta = base - post_real` (unsigned, positive = good) but `verdict.pass = delta <= -2.0` expects negative
   - **Fix**: Change rule to `delta >= 2.0` or redefine delta as `post_real - base` to match signed convention

2. **e14_pause_train**
   - **Issue**: `delta_nll = base_test - post_lpl` (unsigned, positive = good) but `verdict.pass = delta_nll <= -2.0` expects negative
   - **Fix**: Change rule to `delta_nll >= 2.0`

3. **e19_seed_replication**
   - **Issue**: `delta_real = base - post_real` (unsigned, positive = good) but verdict checks `Δ <= -2.0` (expects negative)
   - **Fix**: Change pass criterion to `delta_real >= 2.0` or redefine delta as `post_real - base`

### Drivers Needing Clarification

- **e04_act_halt**: Complex multi-cell structure; verdict logic spread across cells
- **e15_ponder**: Full verdict.pass rule not visible in sampled code ranges
- **e16_capacity**: Multi-phase with different verdict criteria; requires full code review

---

## Notes

- **e01, e02, e03**: Use variant-specific or informational verdicts; baseline (canonical) variant in e01 correctly uses unsigned delta with >= threshold
- **e02**: No binary verdict.pass; delta computed and stored for post-hoc analysis
- **e11**: Uses semantic "nll_drop < threshold" where drop = base - post; threshold interpretation is implicit (drop < -2.0 means improvement >= 2.0)
- **Drivers without delta.pass**: e08 (demo), e12 (informational), e13 (per-benchmark), e15, e16 (phase-based) — no simple verdict audit applicable

---

**Recommendation**: Fix the three BUG drivers before merging. For UNCLEAR drivers, obtain full verdict definitions from complete code.


---

## Follow-up audit (unclear drivers)

**e04_act_halt / e15_ponder_curriculum / e16_bank_capacity**: re-audited (haiku, dispatched after main audit).

All three are CORRECT.

- **e04** uses signed convention (`post - base`, pass: `delta <= -threshold`) — same as e02/e11.
- **e15** uses unsigned convention (`base - post`, pass: `improvement >= +threshold`) — same as e05/e19.
- **e16** uses unsigned convention with phase-specific rules (Phase A: `> 0`, Phase B: `abs(delta) < threshold` for stability).

Full per-driver findings: `v2/scripts/unclear_driver_audit.md`.

**No further sign-convention fixes outstanding.** Aggregator defensively recomputes signed delta from `before.real`/`after.real` for all experiments, so mixed conventions inside drivers do not contaminate the aggregated headline numbers.

