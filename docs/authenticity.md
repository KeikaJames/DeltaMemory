# Authenticity Contract (v0.5)

Every result artifact produced under `experiments/` MUST satisfy this
contract. The contract is the line between "industrial evidence" and
"toy demo": violating it invalidates the artifact for inclusion in any
report, paper, PR description, or wiki page.

This file is normative. It is consumed by `tools/check_authenticity.py`,
which is run manually before opening a PR and (post G+2) by CI.

## 1. Required fields per experiment directory

For every directory `experiments/<phase>/` that contains a result file
(any of: `cells.jsonl`, `cells_smoke.jsonl`, `summary.json`,
`pareto.json`, `verdicts.json`, or any sibling result), there MUST exist
a sibling `env.json` with at least the following fields:

| Field                 | Type   | Description                                        |
|-----------------------|--------|----------------------------------------------------|
| `commit`              | str    | `git rev-parse HEAD` at run time, full sha         |
| `dirty`               | bool   | `git status --porcelain` was non-empty             |
| `dirty_diff_sha1`     | str    | sha1 of `git diff` if dirty, else "0" * 40         |
| `prereg_version`      | str    | e.g. `"W6.v1"`, `"A.v1"`, `"L.v1"`                 |
| `dataset_sha1`        | str OR | sha1 of the primary dataset file used              |
|                       | obj    | OR `{path: sha1}` for multi-dataset experiments    |
| `torch`               | str    | `torch.__version__`                                |
| `transformers`        | str    | `transformers.__version__`                         |
| `python`              | str    | `platform.python_version()`                        |
| `device`              | str    | `cpu` / `mps` / `cuda` / `cuda:0` ...              |
| `dtype`               | str    | `bfloat16` / `float16` / `float32`                 |
| `started_at`          | str    | ISO 8601 UTC                                       |
| `host`                | str    | `socket.gethostname()`                             |

Optional but recommended:

| Field            | Description                                               |
|------------------|-----------------------------------------------------------|
| `cuda`           | `torch.version.cuda` if CUDA build                        |
| `mps_available`  | `torch.backends.mps.is_available()`                       |
| `gpu_name`       | `torch.cuda.get_device_name(0)` if CUDA                   |
| `numpy`          | `numpy.__version__`                                       |
| `seed_global`    | RNG seed at process start                                 |
| `cli_argv`       | full argv used to invoke the run                          |

`env.json` MUST be written at the START of the run, not the end. If a
run crashes mid-way the env is still recoverable.

## 2. Per-cell raw output retention

Aggregate-only artifacts are forbidden. Any `summary.json` or
`REPORT.md` published from an experiment MUST be derivable from a
sibling `cells.jsonl` (or `cells_<arm>.jsonl`) preserved on disk in the
same commit. The aggregate is a pure function of the rows.

A cell row MUST include:
- a `cell_id` (sha1 of canonical cell key)
- the input identifiers (model, method, alpha, seed, prompt_id, ...)
- the primitive measurement(s) (nll, kl, drift, recall, residual_norm, ...)
- a `dataset_sha1` field copying the env.json value (so a row stranded
  from its directory is still self-describing)

Aggregate scripts MUST refuse to run if the underlying cells file is
missing or shorter than the declared expected_n. If a partial cell file
is intentional (interrupted run, resume) the aggregate emits a partial
summary with `partial: true` and `n_observed / n_expected` recorded.

## 3. Bit-equality witness for every injection site

Every injector MUST have an alpha=0 bit-equality test that produces a
witness row in the same `cells.jsonl` with `drift = 0.0` (exact zero,
not "small"). The witness covers the injector's full code path at
alpha=0.

Adding a new injector or new injection site without this witness is a
contract violation. The CI tool flags any experiment whose cells file
contains alpha>0 rows without a corresponding alpha=0 witness for the
same (model, method) pair.

## 4. No fabrication, no editorial trimming

Generation transcripts (Phase Q, marathon checkpoints) are committed
verbatim as produced by the model. Where a report quotes a trimmed
excerpt for narrative purposes:
- the trimmed quote is paired with a link to the full transcript at
  `transcripts/qualitative/<id>.md` or `transcripts/marathon/<run_id>/`
- the full transcript file is in the same commit

The trimmed copy NEVER replaces the full transcript. Removing the full
transcript and keeping only the curated excerpt is a contract violation.

## 5. Cross-machine reproducibility (Phase D)

At least one phase result MUST be reproduced on a second machine with
the same `(commit, dirty=false, dataset_sha1, prereg_version)` triple.
The cross-machine witness lives at `experiments/D_gb10_smoke/witness/`
with one row per `(machine_id, hash(nll_new_per_token))` pair. The diff
between machines is bounded explicitly in
`experiments/D_gb10_smoke/REPORT.md`.

## 6. No hidden seeds

Every stochastic operation records its seed in env.json or in the cell
row. Any seed picked from the system RNG (e.g. `torch.seed()` with no
argument) is forbidden in published runs. RNG state at branch points
(e.g. before sampling generation continuations) is captured.

## 7. Hardware honesty

`device`, `dtype`, `torch`, `transformers`, `python` are recorded
exactly. Mixed-hardware aggregates (e.g. some cells on MPS, some on
CUDA) are flagged in REPORT.md and disaggregated in the cells file.

## 8. Grandfathering

The contract applies prospectively from commit signing this file
onward. Existing artifacts in `experiments/W4_caa_baseline/`,
`experiments/W6_counter_prior/` (smoke), `experiments/A1_rope_audit/`,
`experiments/A3_layer_weighting/`, `experiments/A4_gate_metric/`,
`experiments/A5_profiler_n100/` are grandfathered. The grandfathering
list is in `tools/check_authenticity.py` (`GRANDFATHERED_DIRS`); any
re-run of those experiments writes a new env.json that fully complies.

## 9. Enforcement

`python tools/check_authenticity.py` is the canonical check. Modes:

- (default) walks `experiments/`, prints violations, exits non-zero on
  any violation in non-grandfathered directories.
- `--strict` errors on grandfathered directories too.
- `--paths X Y Z` checks only the listed directories.
- `--bit-equality` additionally enforces section 3 across all cells
  files.

The check is intended to run in CI once the next CI cycle lands; until
then it is run manually by the author before opening a PR. PRs that
break the check must justify in the PR description.

## 10. Why this exists

A research repo without authenticity guards drifts toward selective
reporting: aggregate-only summaries replace raw cells, "cleaned"
transcripts replace what the model said, mixed-hardware numbers get
quietly averaged, post-hoc seeds get picked. Each individual lapse is
plausibly innocent; the cumulative effect is unfalsifiable claims.

This contract makes lapses detectable by a script. That is the entire
point.
