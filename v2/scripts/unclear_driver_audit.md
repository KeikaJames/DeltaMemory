# Audit: Delta-Sign Convention Correctness for Three Flagged Drivers

**Date**: Audit completed for v2/experiments drivers flagged UNCLEAR in prior verdict_sign_audit.md.

**Scope**: Three drivers previously marked UNCLEAR:
1. v2/experiments/e04_act_halt/run.py
2. v2/experiments/e15_ponder/run.py
3. v2/experiments/e16_capacity/run.py

**Convention Standard**:
- v2 standard: `delta_signed = post - base` (negative = improvement)
- Pass rule: `delta_signed <= -threshold` (requires sufficient negative improvement)
- Reference fixes: commits 08b2a304, 15eb718c (e09, e14, e19 corrected)

---

## DRIVER 1: e04_act_halt/run.py

### 1. Delta Formula
**Lines 451**: `delta_nll = post_nll - base_nll`

Computes signed delta using standard v2 convention (post - base).

### 2. Pass/Verdict Rule
**Lines 458–459**:
```python
cell_pass = (post_halts <= 4.0) and (delta_nll <= -2.0)
```
**Line 476**: Stored as `"pass_rule": "post_halts <= 4 AND delta_nll <= -2.0"`

Pass requires `delta_nll <= -2.0`, which correctly gates on negative (improving) delta with threshold -2.0.

### 3. Interpretation Text Consistency

**Pass Branch (pass=true)**:
- Line 453: `"AFTER: post_nll={post_nll:.4f}, post_halts={post_halts:.2f}, Δ={delta_nll:.4f}"` — correctly prints negative improvement
- Lines 651–655 (summary): Prints best cell delta; negative values treated as improvement

**Fail Branch (pass=false)**:
- Implicit in condition `delta_nll > -2.0` — not sufficient improvement
- Summary correctly reports all cells regardless of pass status

**Overall**: Both branches interpret negative delta as improvement consistently with the pass rule.

### 4. Verdict
**CORRECT** ✓

Follows v2 standard exactly:
- Formula: `post - base` (signed) ✓
- Rule: `delta_nll <= -2.0` (negative threshold) ✓
- Interpretation: consistent across both branches ✓

**No fix required.**

---

## DRIVER 2: e15_ponder/run.py

### 1. Delta Formula
**Line 252**: `delta = base_nll - nll`

Computes unsigned delta using alternative convention (base - post, positive = improvement). This differs from v2 standard but matches e05, e11, e19 pattern.

### 2. Pass/Verdict Rule
**Lines 276–277**:
```python
passes = improvement_over_k2 >= 0.3
```
Where `improvement_over_k2 = best_k_gt2_delta - k2_delta` (line 274).

**Line 285**: `"criterion": "improvement_over_k2 >= 0.3"`

Pass requires `improvement_over_k2 >= 0.3`, using positive threshold for positive-valued delta differences.

### 3. Interpretation Text Consistency

**Pass Branch (passes=true)**:
- Lines 317–320 (summary print):
  ```python
  print(f"  K=2 Δ: {k2_delta:.4f}")
  print(f"  Best K>2: {best_k_gt2[0]} with Δ={best_k_gt2_delta:.4f}")
  print(f"  Improvement over K=2: {improvement_over_k2:.4f}")
  print(f"  Passes: {passes} (criterion: ≥0.3 improvement)")
  ```
  Correctly interprets positive improvement values; criterion ≥0.3 matches unsigned delta convention.
  
- Line 323: `return 0 if passes else 1` — returns success (0) when passes=true

**Fail Branch (passes=false)**:
- Implicit in condition `improvement_over_k2 < 0.3` — insufficient improvement
- Symmetric logic applies

**Overall**: Both branches consistently treat positive improvement values as better; pass rule gates on `>= 0.3` (positive threshold), which is correct for unsigned delta.

### 4. Verdict
**CORRECT** ✓

Uses unsigned delta convention (base - post) consistently:
- Formula: `base - nll` (unsigned, positive = improvement) ✓
- Rule: `improvement >= 0.3` (positive threshold) ✓
- Interpretation: consistent across both branches ✓

This is an alternative but valid convention (also used by e05, e11, e19). Not inconsistent with v2 standard; just uses different sign convention. All internal logic is correct.

**No fix required.**

---

## DRIVER 3: e16_capacity/run.py

### 1. Delta Formula

**Phase A (Scaling), Lines 234–235**:
```python
delta_in = base_in - nll_in
delta_out = base_out - nll_out
```

**Phase B (Forgetting), Lines 354, 383, 388, 397**:
```python
delta_A_initial = base_A - nll_A_initial       # Line 354
delta_A_after = base_A - nll_A_after           # Line 383
delta_B = base_B - nll_B                       # Line 388
delta_A_zero = base_A - nll_A_zero             # Line 397
```

All deltas use unsigned convention (base - post, positive = improvement).

### 2. Pass/Verdict Rule

**Phase A, Line 266**:
```python
"pass": all(r["delta_in"] > 0 for r in results) and all(abs(r["delta_out"]) < 2.0 for r in results),
```
Rule: `delta_in > 0` (positive improvement) AND `|delta_out| < 2.0` (small deviation, magnitude-bounded).

**Phase B, Lines 401–402**:
```python
"pass": abs(delta_A_after - delta_A_zero) < 0.3 and abs(delta_B - delta_A_initial) < 1.0,
"rule": "|Δ_A_after - Δ_zero| < 0.3 AND |Δ_B - Δ_A_initial| < 1.0",
```
Rule: Forgetting check (delta_A_after should be near delta_A_zero, i.e., forgotten) AND set_B should reproduce set_A's improvement.

### 3. Interpretation Text Consistency

**Phase A, Lines 237–238**:
```python
f"Δ_in={delta_in:.4f} Δ_out={delta_out:.4f}"
```
Prints raw unsigned delta values; positive = improvement. Consistent with rule `delta_in > 0`.

**Phase B, Lines 355, 384, 389, 398**:
```python
print(f"[e16:forgetting] Δ_A_initial={delta_A_initial:.4f}")
print(f"[e16:forgetting] Δ_A_after_evict={delta_A_after:.4f} (expect ~0, forgotten)")
print(f"[e16:forgetting] Δ_B={delta_B:.4f} (expect similar to Δ_A_initial)")
print(f"[e16:forgetting] Δ_A_zero (empty bank)={delta_A_zero:.4f}")
```
All prints treat positive values as improvement. Comment "expect ~0" means delta should be small (evicted = baseline, delta ≈ 0). Consistent.

**Verdict logic (Line 406)**:
```python
print(f"[e16:forgetting] verdict: {verdict}")
```
Correctly computes and reports both pass conditions (absolute value thresholds work with unsigned deltas).

**Overall**: Both phases consistently interpret unsigned deltas; pass rules use magnitude checks and positive thresholds appropriate for unsigned convention.

### 4. Verdict
**CORRECT** ✓

Uses unsigned delta convention (base - post) throughout both phases:
- Formula: `base - post` (unsigned) ✓
- Phase A rule: `delta_in > 0` (positive threshold) ✓
- Phase B rule: `|delta_diff| < threshold` (magnitude-bounded comparisons) ✓
- Interpretation: consistent across both pass and fail logic ✓

**No fix required.**

---

## Summary Table

| # | Driver | Delta Formula | Convention | Pass Rule (Lines) | Sign Check | Verdict |
|---|--------|---------------|-----------|-------------------|-----------|---------|
| 1 | e04_act_halt | `post - base` (L451) | Signed (v2 standard) | `<= -2.0` (L458) | ✓ Correct | **CORRECT** |
| 2 | e15_ponder | `base - nll` (L252) | Unsigned (alt. convention) | `>= 0.3` (L277) | ✓ Correct | **CORRECT** |
| 3 | e16_capacity | `base - post` (L234–397) | Unsigned (alt. convention) | Phase A: `> 0` (L266); Phase B: `abs()` (L401) | ✓ Correct | **CORRECT** |

---

## Final Conclusion

**All three drivers are CORRECT. No fixes required.**

### Key Findings

1. **e04_act_halt**: Strictly follows v2 standard signed convention (`post - base`, `<= -threshold`).

2. **e15_ponder**: Uses alternative unsigned convention (`base - post`, `>= +threshold`), internally consistent. This convention is also used by e05, e11, e19.

3. **e16_capacity**: Uses unsigned convention throughout both phases (scaling and forgetting), with magnitude-bounded pass rules appropriate for comparing unsigned deltas.

### Why Previously Marked UNCLEAR

The prior audit marked these UNCLEAR because:
- **e04**: Complex multi-cell structure with per-cell verdicts obscured the central pass rule logic.
- **e15**: Full code visibility needed to confirm the verdict rule (was sampled in prior audit).
- **e16**: Multi-phase structure (scaling + forgetting) with different verdict criteria made summary assessment difficult.

Upon full code inspection, all three drivers correctly implement their respective delta sign conventions with matching pass/verdict rules.

### Recommendation

- **No code changes required** for these three drivers.
- All pass sign checks are correct and do not need the fixes applied to e09, e14, e19.
- Document that e04 follows v2 standard (signed); e15 and e16 use valid alternative unsigned convention.

---

**Report Status**: Complete | **Date**: Immediate | **Reviewed**: Three drivers | **Changes Needed**: 0
