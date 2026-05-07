# ATB Validation v1

End-to-end empirical validation of the **AttnNativeBank** (ATB) mechanism, the
sole headline contribution of the RCV-HC paper. Six pre-registered experiments,
all run with SCAR / CAA / LOPI-skip-list explicitly **disabled** so that any
measured effect is attributable to ATB itself.

## Layout

```
experiments/atb_validation_v1/
├── _lib/                       # shared harness (Variant, runners, manifest, aggregator)
├── exp1_core_ablation/         # 5 variants — does pre-RoPE bank read alone work?
├── exp2_position_invariance/   # pre-RoPE vs post-RoPE × position_delta
├── exp3_bit_equal/             # α=0 ⇒ torch.equal across 3 model classes
├── exp4_cf1k_main/             # CF-1k headline result
├── exp5_alpha_sweep/           # 14-point α curve
├── exp6_negative_controls/     # 5 K/V scrambles
├── dispatchers/                # spark1 (GB10) shell entry points
└── SUMMARY.csv                 # cross-experiment canonical table (gitignored)
```

Each `exp<N>_*/` directory contains:
- `PREREG.md` — hypothesis, variants, metrics, acceptance gates, stop conditions
- `run.py` — thin entry point invoking `_lib`
- After a run: `manifest.yaml`, `results.jsonl`, `summary.csv`, `tables/`,
  `plots/`, `run.log` (all gitignored)

## Canonical metric definitions

| name | definition |
|---|---|
| `recall@1` | argmax of next-token logits at end of read prompt equals first token of `target_new` |
| `margin` | Σ logp(target_new \| prompt) − Σ logp(target_true \| prompt), summed over target tokens |
| `target_rank` | 0-indexed rank of first `target_new` token in next-token distribution |
| `js_drift` | mean symmetric Jensen-Shannon divergence (nats) over last-8 logits across 100 fixed neutral prompts |
| `kl_drift` | mean KL(p_baseline ‖ p_patched) (nats) over the same neutral set |
| `bank_attention_mass` | sum of merged-softmax weights on bank columns, mean across (B,H,T,layers) |
| `max_bank_prob` | max merged-softmax weight on any bank column, mean across (B,H,T,layers) |

## Modules
- **enabled**: `AttnNativeBank` only.
- **disabled**: `SCAR`, `CAA`, `LOPI-skip-list`. Recorded in every `manifest.yaml`.

## Running on spark1 (GB10)

```bash
# Cheapest first — bit-equality probe across 3 models.
bash experiments/atb_validation_v1/dispatchers/dispatch_exp3_spark.sh

# Then in any order; each writes under experiments/atb_validation_v1/<exp>/run_<ts>/.
bash experiments/atb_validation_v1/dispatchers/dispatch_exp1_spark.sh
bash experiments/atb_validation_v1/dispatchers/dispatch_exp2_spark.sh
bash experiments/atb_validation_v1/dispatchers/dispatch_exp4_spark.sh
bash experiments/atb_validation_v1/dispatchers/dispatch_exp5_spark.sh
bash experiments/atb_validation_v1/dispatchers/dispatch_exp6_spark.sh
```

## Local dry run (CPU)

```bash
python experiments/atb_validation_v1/exp1_core_ablation/run.py \
    --model google/gemma-3-270m --dtype fp32 --device cpu \
    --seeds 0 --n-prompts 4 \
    --out /tmp/atb_v1_dryrun/exp1
```
The `_lib` harness is fully CPU-compatible for schema validation; production
numbers come only from spark1 runs.

## Reproducibility

Every run writes `manifest.yaml` containing:
- repo git SHA + dirty flag
- dataset path + sha1
- model, dtype, attention_impl
- variant configs, seeds
- enabled / disabled module list
- GPU model + driver version (when CUDA)
- inline definitions of every metric column

Outputs live under `experiments/atb_validation_v1/`, **not** under `runs/`, so
`tests/test_run_authenticity.py` is unaffected.
