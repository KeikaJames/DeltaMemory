# V2 EXPERIMENT DRIVER DELTA-SIGN CONVENTION AUDIT — PASS 3

**Audit Date**: 2024 (Pass 3 — re-audit following missed bugs in prior passes)  
**Target**: Exhaustive delta-sign convention audit for all v2 experiment drivers  
**Prior bugs missed**: e11, e17, e18 (fixed in commits 15eb718c, 97f857ea, 6fb37bcd)

---

## METHODOLOGY

Each driver is audited for **sign consistency** between:

1. **Delta Formula**: How the improvement metric is computed.
   - Standard v2: `delta_signed = post - base` (negative = improvement)
   - Alternative (acceptable if consistent): `delta_unsigned = base - post` (positive = improvement)

2. **Pass/Verdict Rule**: The threshold logic for declaring pass/fail.
   - Standard: `pass if delta_signed <= -threshold`
   - Alternative: `pass if delta_unsigned >= +threshold`

**BUG Definition**: Formula and rule directions **disagree**.  
- E.g., `delta = base - post` (positive for improvement) but `pass if delta <= -threshold` (negative for improvement).

---

## AUDIT RESULTS

### DRIVERS TO AUDIT (NEW, NO RESULTS YET EXAMINED)

---

### **e03_capability_drift/run.py**

**FILE**: EXISTS ✓

**DELTA FORMULA**:
- Line 141-142:
  ```python
  drift_off = nll_off - nll_base
  drift_on = nll_on - nll_base
  ```
- Convention: `post - base` ✓

**PASS/VERDICT RULE**:
- Lines 156-157:
  ```python
  "off_bit_equal": abs(drift_off) <= 0.001,
  "on_within_5pct": abs(rel_drift_on) <= 0.05,
  ```
- Uses **absolute value** (magnitude), not directional. Acceptable for deviation-from-zero tests. ✓

**VERDICT**: ✅ **CORRECT**  
Sign convention is consistent. No bug.

---

### **e04_act_halt_kmax/run.py**

**FILE**: NO FILE (directory empty)  
**VERDICT**: N/A

---

### **e06_relation_disjoint_ood/run.py**

**FILE**: EXISTS ✓

**DELTA FORMULA**:
- Lines 248-249:
  ```python
  delta_train = post_train_real - base_train
  delta_test_ood = post_test_real - base_test
  ```
- Convention: `post - base` ✓

**PASS/VERDICT RULE**:
- Line 251:
  ```python
  "pass": delta_test_ood <= -1.0,
  ```
- Expects negative delta for pass ✓

**VERDICT**: ✅ **CORRECT**  
Formula (post - base) and rule (delta <= -1.0) are sign-consistent.

---

### **e07_per_layer_kproj/run.py**

**FILE**: NO FILE (does not exist; only `e07_perlayer_kproj` exists)  
**VERDICT**: N/A

---

### **e07_perlayer_kproj/run.py**

**FILE**: EXISTS ✓

**DELTA FORMULA**:
- Lines 278-280:
  ```python
  delta_real = post_real - base
  delta_zero = post_zero - base
  delta_off = post_off - base
  ```
- Convention: `post - base` ✓

**PASS/VERDICT RULE**:
- Line 396:
  ```python
  "pass": improvement <= -0.5,
  ```
- Where `improvement = delta_triple - delta_single` (line 391)
- Expects negative improvement (more negative = better) ✓

**VERDICT**: ✅ **CORRECT**  
Sign-consistent. Negative delta means improvement; pass requires negative improvement.

---

### **e08_interrupt_api_demo/run.py**

**FILE**: EXISTS ✓  
**TYPE**: Demo/smoke test (not a standard experiment with pass/fail on improvement)  
**VERDICT**: N/A

---

### **e09_v1_resurrect_attn_native_bank/run.py**

**FILE**: NO FILE  
**VERDICT**: N/A

---

### **e11_dual_channel/run.py**

**FILE**: NO FILE (directory empty)  
**VERDICT**: N/A

---

### **e12_long_short_coexistence/run.py**

**FILE**: NO FILE (directory empty)  
**VERDICT**: N/A

---

### **e12_LT_ST_coexist/run.py**

**FILE**: EXISTS ✓

**DELTA FORMULA** ❌ **BUG FOUND**:
- Lines 263, 290, 297, 303:
  ```python
  LT_only_delta = LT_only_base - LT_only_lpl           # ❌ WRONG
  ST_only_delta = ST_only_base - ST_only_lpl           # ❌ WRONG
  LT_ST_LT_delta = LT_ST_LT_base - LT_ST_LT_lpl         # ❌ WRONG
  LT_ST_ST_delta = LT_ST_ST_base - LT_ST_ST_lpl         # ❌ WRONG
  ```
- Convention: `base - post` (INVERTED) ❌
- Problem: When lpl improves (lower NLL), delta is **POSITIVE** (since base > lpl)

**PASS/VERDICT RULE**:
- Lines 310-312:
  ```python
  pass_LT_only = LT_only_delta <= -1.5
  pass_ST_works = LT_ST_ST_delta <= -1.0
  pass_no_interference = abs(LT_ST_LT_delta - LT_only_delta) <= 0.3
  ```
- Expects negative delta for pass ❌
- But with formula `base - post`, improvements produce **POSITIVE** delta
- **The pass rule will NEVER trigger on real improvements** ❌

**VERDICT**: ❌ **SIGN-INVERTED BUG**

**FIX** (exact edit):
```python
# Line 263: change from
LT_only_delta = LT_only_base - LT_only_lpl
# to
LT_only_delta = LT_only_lpl - LT_only_base

# Line 290: change from
ST_only_delta = ST_only_base - ST_only_lpl
# to
ST_only_delta = ST_only_lpl - ST_only_base

# Line 297: change from
LT_ST_LT_delta = LT_ST_LT_base - LT_ST_LT_lpl
# to
LT_ST_LT_delta = LT_ST_LT_lpl - LT_ST_LT_base

# Line 303: change from
LT_ST_ST_delta = LT_ST_ST_base - LT_ST_ST_lpl
# to
LT_ST_ST_delta = LT_ST_ST_lpl - LT_ST_ST_base
```

---

### **e14_pause_head_train/run.py**

**FILE**: EXISTS ✓

**DELTA FORMULA**:
- Line 338:
  ```python
  delta_nll = post_lpl - base_test
  ```
- Convention: `post - base` ✓

**PASS/VERDICT RULE**:
- Lines 340-341:
  ```python
  "pass": delta_nll <= -2.0 and post_pause_stats["mean_pauses"] <= 8.0,
  "rule": "Δ NLL ≤ -2.0 AND mean_pauses ≤ 8 (Δ = post - base; negative = improvement)",
  ```
- Expects negative delta for pass ✓
- **Explicitly documents the convention** ✓

**VERDICT**: ✅ **CORRECT**  
Sign-consistent. Formula and rule both follow v2 standard (post - base).

---

### **e16_bank_capacity_forgetting/run.py**

**FILE**: NO FILE  
**VERDICT**: N/A

---

### **e18_chained_2hop/run.py**

**FILE**: NO FILE  
**VERDICT**: N/A

---

## VERIFICATION OF PRIOR FIXES

---

### **e11_noise_robustness/run.py** (fixed in commit 15eb718c)

**FILE**: EXISTS ✓

**DELTA FORMULA** (line 389):
```python
delta_signed = post_real - base
```
- Convention: `post - base` ✓

**PASS/VERDICT LOGIC** (lines 392-423):
```python
helped = delta_signed <= -threshold  # noise/degenerate bank reduced NLL by >= 2
verdict["pass"] = not helped         # PASS means: noise did NOT help
```

**Analysis**:
- The "pass" criterion inverts the "helped" boolean
- This is **intentional**: the test asks "did noise/degenerate bank help?"
- For pass: "noise did NOT help" → `not helped` → pass ✓

**VERDICT**: ✅ **CORRECT**  
Prior fix is still consistent. No regression.

---

### **e14_pause_train/run.py** (was fixed)

**FILE**: EXISTS ✓

**DELTA FORMULA** (line 338):
```python
delta_nll = post_lpl - base_test
```
- Convention: `post - base` ✓

**PASS/VERDICT RULE** (lines 340-341):
```python
"pass": delta_nll <= -2.0 and post_pause_stats["mean_pauses"] <= 8.0,
"rule": "Δ NLL ≤ -2.0 AND mean_pauses ≤ 8 (Δ = post - base; negative = improvement)",
```

**VERDICT**: ✅ **CORRECT**  
Prior fix is consistent. Explicit documentation of convention.

---

### **e17_negation_robustness/run.py** (fixed in commit 97f857ea)

**FILE**: EXISTS ✓

**DELTA FORMULA** (lines 256, 264, 276, 288):
```python
delta_a = lpl_a - base_a
delta_b = lpl_b - base_b
delta_c = lpl_c - base_c
delta_d = lpl_d - base_d
```
- Convention: `post - base` ✓

**PASS/VERDICT LOGIC** (lines 299-320):
```python
if delta_a <= -1.0: pass         # (a) standard/standard
if delta_b > -0.5: pass          # (b) standard/random
if delta_c > -0.5: pass          # (c) negated/random
if delta_d <= -1.0: evidence     # (d) negated/target_true
```

**Analysis**:
- Different thresholds and directions are **intentional** per test design
- (b) and (c): `delta > -0.5` means bank did NOT substantially help random targets → pass ✓
- (a) and (d): `delta <= -1.0` means bank did help standard targets → pass ✓
- Internally consistent per design

**VERDICT**: ✅ **CORRECT**  
Prior fix is still consistent. Design is intentionally asymmetric.

---

### **e18_2hop/run.py** (fixed in commit 6fb37bcd)

**FILE**: EXISTS ✓

**DELTA FORMULA** (lines 388-390):
```python
delta_AB_vs_None = post_results["AB_both_two_hop"] - post_results["None_two_hop"]
delta_AB_vs_A = post_results["AB_both_two_hop"] - post_results["A_only_two_hop"]
delta_AB_vs_B = post_results["AB_both_two_hop"] - post_results["B_only_two_hop"]
```
- Convention: `post - base` ✓

**PASS/VERDICT RULE** (line 392):
```python
pass_criterion = (delta_AB_vs_A <= -0.8 and delta_AB_vs_B <= -0.8)
```
- Expects negative delta for pass (AB helps) ✓

**VERDICT**: ✅ **CORRECT**  
Prior fix is still consistent. Formula and rule sign-aligned.

---

### **e19_seed_replication/run.py** (was fixed)

**FILE**: EXISTS ✓

**DELTA FORMULA** (line 213):
```python
delta_real = post_real - base
```
- Convention: `post - base` ✓

**PASS/VERDICT RULE** (lines 286, 299):
```python
all_beat_thresh = all(d <= thresh for d in deltas)  # thresh = -2.0 or -3.0
pass = all_beat_thresh and std_pass
```
- Expects negative thresholds for pass ✓

**VERDICT**: ✅ **CORRECT**  
Prior fix is consistent.

---

## SUMMARY

| Driver | File | Result | Issue |
|--------|------|--------|-------|
| e03_capability_drift | ✓ | ✅ CORRECT | None |
| e04_act_halt_kmax | ✗ | N/A | No file |
| e06_relation_disjoint_ood | ✓ | ✅ CORRECT | None |
| e07_per_layer_kproj | ✗ | N/A | No file |
| e07_perlayer_kproj | ✓ | ✅ CORRECT | None |
| e08_interrupt_api_demo | ✓ | N/A | Demo only |
| e09_v1_resurrect_attn_native_bank | ✗ | N/A | No file |
| e11_dual_channel | ✗ | N/A | No file |
| e12_long_short_coexistence | ✗ | N/A | No file |
| **e12_LT_ST_coexist** | ✓ | ❌ **SIGN-INVERTED** | **BUG FOUND** |
| e14_pause_head_train | ✓ | ✅ CORRECT | None |
| e16_bank_capacity_forgetting | ✗ | N/A | No file |
| e18_chained_2hop | ✗ | N/A | No file |

**Prior Fixes (Verified)**:
- e11_noise_robustness: ✅ CORRECT
- e14_pause_train: ✅ CORRECT
- e17_negation_robustness: ✅ CORRECT
- e18_2hop: ✅ CORRECT
- e19_seed_replication: ✅ CORRECT

---

## BUGS FOUND THIS PASS

### **1. e12_LT_ST_coexist/run.py — SIGN-INVERTED DELTA**

**Severity**: HIGH (Pass rule will never trigger on real improvements)

**Lines Affected**: 263, 290, 297, 303

**Current Code**:
```python
LT_only_delta = LT_only_base - LT_only_lpl
ST_only_delta = ST_only_base - ST_only_lpl
LT_ST_LT_delta = LT_ST_LT_base - LT_ST_LT_lpl
LT_ST_ST_delta = LT_ST_ST_base - LT_ST_ST_lpl
```

**Problem**:
- When LPL improves (lower NLL), delta is POSITIVE
- Pass rules (lines 310-312) require delta <= -threshold
- Result: Pass rules never trigger on improvements ❌

**Fix**:
```python
LT_only_delta = LT_only_lpl - LT_only_base
ST_only_delta = ST_only_lpl - ST_only_base
LT_ST_LT_delta = LT_ST_LT_lpl - LT_ST_LT_base
LT_ST_ST_delta = LT_ST_ST_lpl - LT_ST_ST_base
```

**Commit Recommendation**: Label as bugfix, associate with delta-sign convention standardization.

---

## RECOMMENDATIONS

1. **Immediate**: Apply fix to e12_LT_ST_coexist (lines 263, 290, 297, 303).
2. **Documentation**: Add delta convention comment to all drivers similar to e14_pause_train line 341.
3. **Testing**: Add regression test checking sign consistency of delta formula and pass rules.
4. **Code Review**: Check if any drivers are using unusual conventions (e.g., positive for improvement) that might be misinterpreted as bugs.

