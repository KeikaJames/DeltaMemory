# W.4 — CAA-injection paired comparison · Pre-Registration

**Status**: PREREG (frozen before any cells.jsonl row is written)
**Owner**: project lead
**Predecessor**: W.3 DECISION (CAA promoted to main-line candidate after LOPI W.2 FAIL)
**Hypothesis under test**:

> H4 — CAA residual-stream injection significantly reduces drift and/or
> increases counter-prior lift relative to (a) no-memory and (b) the LOPI
> shipped default, paired by (model × α × prompt × seed).

---

## 1. Methodology paradigm

**Controlled variable + Benchmarking** — single mechanism axis (none / LOPI / CAA),
all other variables held fixed.

---

## 2. Grid

| Axis | Levels | Count |
|---|---|---|
| Model | gpt2-medium, Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, google/gemma-3-270m, gemma-3-1b-it (substitute for E2B if E2B unavailable on 64 GB) | 5 |
| Method | `none` (bank only), `lopi_default` (full LOPI A3), `caa` (X.3 default config) | 3 |
| α | 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 | 7 |
| Seed | 0, 1, 2 | 3 |
| Prompts | gold_30prompts.jsonl (sha-locked) | 30 |

Total: 5 × 3 × 7 × 3 × 30 = **9450 forward passes**.

> *Substitution policy*: if a model fails to load on the available device within
> 30 s, substitute the next-larger model from the same family; substitution is
> recorded in REPORT.md and the affected rows are tagged `model_substituted=true`
> in cells.jsonl. **Substitution is not allowed silently.**

---

## 3. Fixed configuration

- V-scale: ON for all non-Gemma models, RMS cap = 0.5 (per W.3 §3.2)
- mHC shield: OFF (this run isolates the injector axis)
- Bank: random_v init, n_slots = 8, dtype = bfloat16
- Device: MPS (Apple Metal); cuda accepted if available
- Eval metric: drift = mean per-token NLL on gold_30prompts.jsonl held-out span
  (last 64 tokens; first 32 tokens used as prompt)
- α=0 row of every (model, method, seed) triple is the bit-equality witness;
  any non-zero `|drift_alpha=0 - drift_no_memory|` aborts that cell

### CAA-specific configuration (X.3 defaults, frozen)

- `inject_layer = "mu_arch"` (resolved via LOPI profiler at run start)
- `use_lopi_gate = False` (vanilla CAA; gated variant deferred to W-T3.4)
- `mode = "additive"`
- Steering-vector calibration: 16 (positive, neutral) prompt pairs from
  `experiments/datasets/caa_calibration_pairs.jsonl` (will be created in this
  PREREG commit)

### LOPI-default configuration (A3 from W.2, frozen)

- ortho = True, gauss = True, deriv = True, full LOPI
- LOPIConfig defaults from `deltamemory.memory.lopi_inject` as of commit
  `5d044870`
- Known-issue carry-forward (per W.2 §"Fix 1–4"): Gaussian centering offset
  for Qwen and γ_t = 1.0 in single-prompt eval. **These are not patched** so
  this run measures shipped LOPI vs CAA.

---

## 4. Statistical analysis (locked)

- Per-cell pairing key: `(model, alpha, seed, prompt_id)`
- Primary test: paired Wilcoxon signed-rank, two-sided, on `drift_method - drift_none`
- Family of tests: 5 models × 7 α × 2 method-vs-baseline contrasts = **70 tests**
- Multiple-comparison correction: Holm–Bonferroni on the full family
- Significance threshold: corrected p < 0.01 ("significantly better")
- Effect-size reporting: median paired difference + bootstrap 95 % CI (B = 1000)

---

## 5. Pre-registered verdict matrix

| Outcome | Decision |
|---|---|
| CAA beats LOPI on ≥3/5 models at ≥3/7 α (Holm-corrected p<0.01) | CAA promoted to default-on; LOPI remains ablation flag (W.3 §3.1 confirmed) |
| CAA beats `none` on ≥3/5 models but ties LOPI | both retained as alternative defaults; user picks per-arch |
| CAA does **not** beat `none` significantly | main-line falls back to "V-scale + raw bank"; W-T3 opens for both LOPI and CAA |
| α=0 bit-equal red-line violated | run aborted, code patched, run restarted |

---

## 6. Anti-tampering

1. PREREG.md committed before cells.jsonl is opened (this commit)
2. cells.jsonl appended one row per cell; no rewrites; first column = `cell_id`
   = `sha1(f"{model}|{method}|{alpha}|{seed}|{prompt_id}")`
3. Aggregate computation lives in `aggregate.py`; raw cells.jsonl is the only
   ground truth — verdict tables in REPORT.md must be re-derivable from cells
4. env.json captured at run start: torch / transformers / commit / dtype /
   device / model SHAs

---

## 7. Out-of-scope (deferred)

- Gated CAA (`use_lopi_gate=True`) → W-T3.4
- Non-additive CAA modes (projected, normalized) → W-T3.4
- Multi-layer CAA injection → W-T3.4
- Counter-prior lift on LAMA-500 → W.6 (this run measures only drift on neutral set)
- MoE adaptation → W.5

---

## 8. Sign-off

PREREG closed at commit time. The next commit on this branch must be the
cells.jsonl run output (or a corrigendum amending §3 with explicit reason).
