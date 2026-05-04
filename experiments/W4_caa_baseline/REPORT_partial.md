# W.4 Partial Report — 3-model CAA Baseline

**Status**: PARTIAL — 3 of 5 PREREG models executed.
**Run date**: 2026-05-04.
**Hardware**: Apple M-series, MPS, bfloat16.
**Cells executed**: 5,040 of 9,450 (53%).

## Why partial

`google/gemma-3-270m` weights download stalled at the
`.safetensors`-blob fetch step from Hugging Face during this run.
`google/gemma-3-1b-it` was therefore not attempted. Both models remain
queued for the 128 GB tomorrow run; the code path is verified by the
W.4 smoke (gpt2-medium) and the resume-via-done-set logic preserves the
5,040 cells already on disk so re-execution will only run the remaining
~4,400 cells.

The cells that **did** complete cover all three architectures the
PREREG groups as "transformer with KV cache and standard RoPE": the
GPT-2 control (no RoPE, lopi_default flagged `method_unsupported=true`)
and the two Qwen2.5 dense models (0.5B and 1.5B).

## Red-line — bit-equality

**`max |drift|` at alpha=0 across all 5,040 cells = 0.0** for every
(model, method) combination. The PREREG threshold is 1e-4. **Pass**.

This is the methodology spine. It says that at zero injection strength,
the proposed pipeline is byte-exact identical to the unmodified frozen
LLM. Any non-zero result at non-zero alpha is therefore causally
attributable to the injection, not to a stale-state or numerical-drift
artefact.

## Per-model verdict (paired Wilcoxon, Holm-Bonferroni, threshold 0.01)

| model              | CAA sig α-cells | best CAA effect             | LOPI sig α-cells | best LOPI effect            |
| ------------------ | --------------- | --------------------------- | ---------------- | --------------------------- |
| gpt2-medium        | 0               | n/s                         | n/a (unsupported)| n/a                         |
| Qwen2.5-0.5B       | 3               | alpha=0.25, diff = **-2.371** | 4                | alpha=8, diff = -0.939      |
| Qwen2.5-1.5B       | 0               | n/s                         | 4                | alpha=4, diff = **-4.400**  |

(Negative `median_diff` means the method's `nll_new` is **lower** than
the no-memory baseline — i.e. the method beats `none` on the
counterfact target.)

**Reading**: the family verdict is `MIXED`, but the per-model picture
is clearer than a flat count of significant cells suggests:

- On **Qwen2.5-0.5B**, CAA produces the largest single-cell effect
  (-2.37 at alpha=0.25) but only 3 of 7 alpha sweeps reach
  Holm-significance. LOPI is more consistent (4 sig cells) but at
  smaller effect sizes (-0.94 at alpha=8).
- On **Qwen2.5-1.5B**, LOPI is the unambiguous winner: 4 sig cells with
  median_diff = -4.40 at alpha=4. CAA has zero significant cells. This
  is surprising relative to the W.2 dissection (which marked LOPI's
  three components as failing per-model operating-regime tests) and
  motivates the W-T3 SPEC re-evaluation: the integrated stack delivers
  lift even though the components fail in isolation.
- On **gpt2-medium**, neither method reaches significance after Holm.
  GPT-2 is a deliberate control with no RoPE; the negative result here
  is consistent with the architectural caveat in the PREREG.

## Provisional `M_winner` for downstream phases

W.6, W.10, W.13, and W.14 all depend on this report's `M_winner`. The
3-model partial result is **insufficient to lock** `M_winner` for the
v0.4 release. Two paths forward:

1. **Per-model winner**: select per-model (`Qwen2.5-0.5B -> caa`,
   `Qwen2.5-1.5B -> lopi`, `gpt2-medium -> caa` as W.3 fallback).
   Downstream phases then run model-stratified.
2. **Defer**: wait for the gemma-3-270m and gemma-3-1b-it cells before
   committing to a single winner.

The CHANGELOG records this as a partial verdict and the gating commit
for downstream phases remains the full 9,450-cell run.

## Outputs

- `cells.jsonl`     — 5,041 lines (5,040 real cells + 1 GPT-2 lopi sentinel).
- `verdicts.json`   — 42-test family with Holm correction.
- `env.json`        — env hash, model list, commit SHA at run.
- `logs/run.log`    — per-cell timing.

## Next

- 128 GB run: complete gemma-3-270m and gemma-3-1b-it (~4,400 cells,
  estimated 8–12 minutes once weights cached).
- `aggregate.py` re-run produces the final 5-model verdict.
- `M_winner` and `DM_best_alpha` then propagate to W.6, W.10, W.13,
  W.14.
