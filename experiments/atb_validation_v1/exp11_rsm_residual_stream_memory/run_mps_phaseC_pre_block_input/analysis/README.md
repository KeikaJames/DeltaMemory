# Exp11 RSM Analysis

**Verdict:** STABILIZER_ONLY

Verdict ladder uses bootstrap CI on `margin`:
- **PASS_STRONG**: correct.ci_lo > max(controls.mean) and correct > base.
- **PASS_DIRECTIONAL**: correct.mean > all controls and correct > base.
- **STABILIZER_ONLY**: correct > base and correct > random_memory.
- **FAIL**: otherwise.

Controls considered: random_memory, shuffled_layers, gate_off, gate_uniform.

See `phase_a_comparison.csv`, `phase_b_comparison.csv`, and per-config summaries.
