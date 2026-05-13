# W.12 Pre-Registration: Full Mechanism Ablation Cross-Product

**Status**: locked.
**Authored**: 2026-05-04.
**Depends on**: W.4 verdict (CAA configuration confirmed); W-T3 SPEC fixes
  for LOPI (Fix 1, 3 minimum required) accepted into trunk; ECOR remains
  opt-in throughout.
**Hardware target**: 128 GB recommended; 64 GB completes 3 of 5 models.

W.12 is the **mechanism-attribution** experiment. The v0.4 method is a
stack: bank + (mHC shield) + (LOPI) + (V-scale) + (CAA) + (ECOR).
Reviewers will ask "which knob matters?" W.12 turns each knob on/off
independently and measures the marginal and interaction effects on a
fixed evaluation grid.

The output of W.12 is the **production default configuration** of v0.4 —
the on/off pattern of the four switchable mechanisms that maximises
`nll_new` lift while keeping `kl_unrel` below threshold.

---

## 1. The four ablatable mechanisms

| flag         | code path                                | default-on | default-off witness |
| ------------ | ---------------------------------------- | ---------- | ------------------- |
| `mhc`        | `deltamemory.memory.mhc_shield`          | off        | shield_attention_weights bypass |
| `lopi`       | `deltamemory.memory.lopi`                | off        | LOPIState.identity  |
| `vscale`     | `deltamemory.memory.v_scale`             | on         | scale=1.0           |
| `caa`        | `deltamemory.memory.caa_injector`        | on (winner)| alpha=0 short-circuit |

ECOR (`ecor.py`) is **NOT** in the cross-product. ECOR is opt-in by
design (W.3 ruling) and its interaction with the other four is studied
in a separate W.12.E follow-up.

The cross-product is therefore `2^4 = 16` distinct method
configurations.

## 2. Grid

`5 models * 16 configs * 5 alphas {0, 0.5, 1, 2, 4} * 3 seeds * 30 prompts
= 36,000 cells`. The 30 prompts are the W.6 counter-prior set (same
prompts so that lift comparisons are paired against W.6 results).

Models: gpt2-medium, Qwen2.5-0.5B, Qwen2.5-1.5B, gemma-3-270m,
gemma-3-1b-it. GPT-2 has no RoPE, so configs containing `lopi=on` flag
`method_unsupported=true` and emit a single sentinel row per
(config, alpha, seed) instead of 30 cells. With this exception, the
non-sentinel cell count is `(5 - 1*8)*16 + 1*8`-corrected per W.4
template.

## 3. Hypotheses

**H12a (production default exists)**:
  Among the 16 configs, there exists a Pareto-best config defined by
  `(median_p nll_new < median_p nll_new(no_memory)) AND
   (median_p kl_unrel < 0.5 nats)` whose 95% bootstrap CI on
  `nll_new lift` does not overlap the second-best config's CI on at
  least 3 of 5 models. PASS = the v0.4 production default is named in
  the report.

**H12b (mHC contribution is non-significant if all else on)**:
  Marginal contribution of `mhc=on` (averaged across the 8 configs that
  pair (mhc=on, *) vs (mhc=off, *)) is < 0.05 nats absolute on every
  model, by paired test on `(seed, prompt_id)`. Holm threshold 0.05.
  If H12b passes, **mhc is recommended for removal** from the v0.4
  default stack (already deprioritised by W.1 / W.3).

**H12c (LOPI x CAA non-redundancy)**:
  Two-way ANOVA on `nll_new` with factors `lopi` and `caa` produces an
  interaction term whose Holm-adjusted p < 0.05 on at least 3 of 5
  models. PASS means the two mechanisms are not redundant
  (cumulative > sum-of-individual). FAIL means one of {LOPI, CAA} can
  be dropped from the default with no expected lift loss.

**H12d (V-scale + mhc are not synergistic)**:
  ANOVA interaction `mhc x vscale` is not significant at p<0.05 on >=3
  of 5 models. This **falsifies** the U.1 conjecture that "mHC and
  V-scale must be used together". Pre-registered as a falsifiable
  predicted-PASS — i.e. we expect H12d to PASS, which is the
  contrarian outcome relative to U.1.

**H12e (alpha=0 invariance over all 16 configs)**:
  At alpha=0, `max_{config} |nll_new(config) - nll_new(no_memory)| <
  1e-4` on all models. Inherited W.4 red-line, applied across all 16
  configs.

## 4. Statistics

- Each (model, alpha) cell yields `(seed, prompt) = 90` paired
  observations. Within model, paired-Wilcoxon for binary contrasts.
- Two-way ANOVA (statsmodels OLS + anova_lm) for H12c/d on the 8
  configs at the W.6-best alpha. ANOVA factor levels are {on, off}
  for each switch; replicate dimension = (seed, prompt).
- Holm-Bonferroni applied per hypothesis family.
- Effect size: median paired diff with bootstrap 95% CI (B=1000,
  seed=0).

## 5. Red-lines and aborts

1. H12e violation -> abort (the bit-equality contract is the spine).
2. ANOVA homogeneity-of-variance Levene test p<0.001 on a model x
   factor pair -> flag and switch that pair to non-parametric
   Friedman test (pre-registered fallback).
3. Memory cap on 64 GB host: gemma-3-1b-it + lopi=on may OOM during
   the LOPI projector preallocation; if RSS > 56 GB at config start,
   skip and tag config_oom=true on a sentinel row.

## 6. Deliverables

- `cells.jsonl`     — 36,000 rows (with sentinels for GPT-2 lopi
                       configs and any OOM skips).
- `summary.json`    — H12a-e verdicts, per-model best config name,
                       16-bar Pareto frontier numerics.
- `figures/`        — 16-bar Pareto (5 figures, one per model),
                       ANOVA heatmap (1 figure), main-effects bars
                       (1 figure).
- `REPORT.md`       — narrative + named v0.4 production default config.
- `env.json`        — env hash; LOPI fix manifest hash (must match
                       trunk SHA at run launch).

## 7. Out of scope

- ECOR (W.12.E follow-up, opt-in only).
- Long-context interaction (W.7 cross-product is a separate W.7.A).
- Multi-fact / multi-turn ablations (W.8.A, W.9.A).

---

End of pre-registration. The 16 config names are frozen as
`{mhc?, lopi?, vscale?, caa?}` boolean tuples; introducing a 17th
mechanism (e.g. ECOR) requires a new W.12.E phase.
