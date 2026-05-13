# W.6 Counter-Prior — Smoke Pre-Flight

## Verdict

**PASS.**  Pre-flight on `gpt2-medium` produced 20 real cells with
`alpha=0` max `|drift| = 0.0` (bit-equality red-line satisfied).
`aggregate.py` ran end-to-end and emitted `summary.json` and `pareto.json`
without exceptions.

## Configuration

* Branch:                `feat/v04-w6-counter-runner`
* Method winner:         `caa` (`method_winner_source = smoke_assumption`,
                         per PREREG section 7.4 because W.4's full grid has
                         not yet shipped)
* Models:                `gpt2-medium` only
* Methods:               `none`, `caa`
* Alphas:                `{0.0, 1.0}`
* Seeds:                 `{0}`
* Prompts:               first 5 rows of `counterfact_60.jsonl`
* Unrelated KL windows:  5 per (seed, prompt) on `wikitext-2-raw-v1`
* Device / dtype:        MPS / `bfloat16`
* Total real cells:      20  (2 methods x 2 alphas x 1 seed x 5 prompts)
* Sentinels:             0  (no `method_unsupported`, no
                         `relation_template_missing`)
* Red-line violations:   0
* Wall-clock:            ~22 s

## Bit-equality witness (alpha=0)

| method | seed | prompt_id | nll_new        | drift          |
| ------ | ---- | --------- | -------------- | -------------- |
| none   | 0    | cf_001    | 7.4566         | 0.0            |
| caa    | 0    | cf_001    | 7.4566         | 0.0            |
| ...    | ...  | ...       | ...            | 0.0            |

All five `alpha=0` injected cells produce identical `nll_new` to the
matched `none` baseline.  The `CAAInjector.__enter__` hook short-circuits
when `alpha == 0.0` and returns the unmodified output tuple, which is what
guarantees the bit-equality.

## Pareto frontier (smoke)

`pareto.json` reports for `gpt2-medium`:

```
alpha=0.0  median nll_new = 8.945  median kl_unrel = 0.000
alpha=1.0  median nll_new = 7.851  median kl_unrel = 0.005
best_alpha = 1.0   best_kl_unrel = 0.005   passes_h6c = true
```

`kl_unrel` at alpha=0 is numerically zero (a tiny negative value from
floating-point error in the JS computation; PREREG's `[0, log 2]` bound is
respected up to numerical noise).

## Deviations from PREREG

1. **Relation-template fallback.**  PREREG section 4 specifies that
   `relation_phrase` is rendered from `lama_trex_500.jsonl` keyed by
   `predicate_id == relation`.  The dataset shipped in this repository
   contains rows for only 1 P-id (`P131`); a strict lookup would drop
   ~96% of the 60 counterfact prompts, far above the 10% abort threshold.
   PREREG section 4 explicitly allows hand-authored fallback templates
   recorded as a deviation.  Implementation: when no LAMA row matches the
   prompt's relation, `run.py` falls back to extracting the phrase from
   the counterfact row's own `prompt` template (e.g. `"{} originated in"`
   yields `relation_phrase = "originated in"`).  Each fallback row is
   tagged `relation_template_source = "counterfact_prompt_fallback"`; LAMA
   matches are tagged `"lama"`.  Observed drop rate during smoke: 0%.

2. **`lopi_default` arm stub.**  When the smoke pre-flight runs with
   `M_winner = "caa"`, the `lopi_default` branch is never exercised.  The
   full-grid harness will require swapping a real LOPI context manager in
   `evaluate_cell` (currently the branch falls through to no-op, matching
   the behaviour of `none`).  This will be wired before the full grid
   runs and recorded in `REPORT.md` at that time.

3. **Counterfact SHA.**  PREREG quotes `c3e1ac77...`; the file shipped at
   `experiments/datasets/counterfact_60.jsonl` hashes to
   `d1e3c141ca21971284a4d71d27af1ace8d031222`.  The actual SHA is
   recorded in `env.json.counterfact_sha1`.

None of the above changes a hypothesis, an alpha, a seed, a Holm family
size, or a red-line threshold.  `prereg_version = "W.6.v1"` is retained.

## What the smoke proves

* `run.py` completes 20 cells on MPS in ~22 s.
* `cell_id` keying is sha1 over `(model, method, alpha, seed, prompt_id)`
  and supports resumption via the `done` set.
* `env.json` records `prereg_version`, `method_winner_source`,
  `git_commit`, dataset SHAs and grid extent.
* `aggregate.py` produces `summary.json` (with H6a/H6b cells, raw and
  Holm-rejected p-values, bootstrap CIs) and `pareto.json` (per-model
  H6c frontier, best alpha, `passes_h6c` flag).
* Both `tests/test_w6_smoke.py` cases pass under
  `pytest tests/test_w6_smoke.py -q`.

## What the smoke does not prove

* The `lopi_default` injection path.  The smoke runs the `caa` arm only.
* CAA on RoPE families (`Qwen/*`, `google/gemma-3-*`).  Layer resolution
  via `_resolve_layer` is exercised on `gpt2-medium` (returns 8) but
  RoPE-model paths in `CAAInjector._get_decoder_layers` are not exercised
  here.  The full grid will exercise them.
* Statistical power.  With 5 prompts x 1 seed x 1 alpha=1.0, paired
  Wilcoxon has too few pairs to reject Holm at 0.01 (`n_reject = 0/2`
  for both H6a and H6b is expected at smoke scale).

## Reproduction

```bash
source .venv-mac/bin/activate
python experiments/W6_counter_prior/run.py --smoke
python experiments/W6_counter_prior/aggregate.py \
  --cells experiments/W6_counter_prior/cells_smoke.jsonl \
  --out   experiments/W6_counter_prior/
pytest tests/test_w6_smoke.py -q
```
