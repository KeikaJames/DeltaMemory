# Exp11 — Residual Stream Memory (RSM) Final Verdict

**Status:** **FAIL — strong-steering, no fact-specific retrieval.**

Model: `Qwen/Qwen3-4B-Instruct-2507` · Device: Apple MPS · dtype: bf16 ·
n=21 prompts × seeds {0,1} · bank=64 · grid: eta ∈ {0.05, 0.10} × theta ∈ {0.30, 0.50}.

## Why we ran Phase C fallbacks

The original `block_output` smoke had three structural red flags:

1. `shuffled_layers > correct_memory` ⇒ layer identity carries no fact-specific structure.
2. `correct_memory ≡ gate_off` ⇒ theta gate never activates; cosine gate is non-discriminative.
3. All RSM variants beat `base_model` by ~3.3 nats ⇒ injection is global steering, not retrieval.

To rule out hook-point as the confound, we added `pre_block_input` (forward
pre-hook on each decoder layer) and `mlp_mid` (forward pre-hook on
`layer.mlp.down_proj`). We also added a stricter control: `gate_uniform`
(all-1 weights for every memory in bank), distinct from `gate_off` which
keeps `clamp_min(score, 0)` weights.

## Three-hook matrix (best config per hook by `gap = correct − max(controls)`)

| hook_point | best (eta, theta) | base | correct | random | shuffled | gate_off | gate_uniform | gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `block_output` (prior) | (0.05, *) | −4.29 | −0.93 | −1.01 | **−0.81** | −0.93 | n/a (pre-existed) | −0.12 |
| `pre_block_input` | (0.10, 0.30) | −4.29 | −0.93 | −1.00 | **−0.88** | −0.93 | −0.90 | −0.05 |
| `mlp_mid` | (0.10, 0.30) | −4.29 | −2.04 | −2.09 | −4.29 | −2.03 | **−1.72** | −0.31 |

`shuffled_layers` margin at `mlp_mid` (−4.29) collapses back to `base_model`,
showing intermediate-dim memories DO carry layer-tied structure — yet
correct vs random is essentially flat (−2.04 vs −2.09), and `gate_uniform`
beats every retrieval-gated variant by ~0.3 nats.

Across all four configurations on each hook, `gap_control < 0`. **No hook
point produces correct_memory > all controls.**

## Why retrieval failed

Per-row diagnostics on `pre_block_input` (best config):

| metric | correct | random | shuffled | gate_uniform |
|---|---:|---:|---:|---:|
| mean cosine score | 0.767 | 0.766 | 0.680 | 0.767 |
| top − mean separation | **0.039** | 0.035 | 0.055 | 0.039 |
| top_memory_hit (rank-1 = correct fact) | 28.6% | 0% | 11.9% | n/a |
| activation_rate (s > theta) | 1.000 | 1.000 | 1.000 | 1.000 |

The cosine *ranking* is non-trivially correct (28.6% rank-1 hit, well above
1/64 ≈ 1.6% chance) — so the residual *does* contain weak fact-identifying
structure. But the score distribution is extremely compressed (top
0.806, min 0.703, std 0.025), so:

- Theta ∈ {0.30, 0.50} never gates anything off → `correct ≡ gate_off` exactly.
- The 0.039-nat separation between top match and the rest is too small for
  the cosine-weighted injection to express retrieval as a margin gain.
- Adding all memories with weight 1 (`gate_uniform`) lands in the same
  steering basin and even slightly outperforms cosine-weighted variants,
  because the residual signal that helps the LM at all is the *bulk
  direction*, not the *picked vector*.

## Verdict ladder applied

`PASS_STRONG` requires `correct.ci_lo > max(controls.mean)`. At the best
hook (`pre_block_input`, eta=0.10, theta=0.30) the 95% bootstrap CI for
correct is [−1.37, −0.49] and `max(controls.mean) = −0.88`. CI lower
bound is below max control mean by ~0.5 nats, so it fails at
`PASS_STRONG`. It also fails `PASS_DIRECTIONAL` because `shuffled_layers`
and `gate_uniform` both have higher mean margin than `correct_memory`.

Final ladder result on every (hook, config) pair: **STABILIZER_ONLY**.
This is the "RSM lifts base but loses to controls" basin — i.e. the
method functions as a generic activation stabilizer/steerer, not a memory
retriever.

## Interpretation (per PREREG)

- `shuffled` ≥ `correct` at block_output / pre_block_input → layer identity
  not a useful key at residual-boundary hooks.
- `gate_uniform` ≥ `correct` at mlp_mid → score gating is uninformative;
  all-memory averaging wins.
- `correct ≡ gate_off` at boundary hooks → cosine threshold never fires;
  signal is too compressed.
- All variants ≫ base → strong steering effect from any residual nudge.

## Out of scope (deliberately not pursued)

Per the user's explicit instruction (don't waste tokens on small knobs),
we did not iterate on:

- Per-layer eta schedules.
- RMS-norm caps on delta.
- Multi-token write prompts or alternate keys.
- Score sharpening (softmax temperature, top-1 hard selection).

If RSM is to be revisited, the natural next step suggested by the
diagnostics is **hard top-1 retrieval with whitening** — discard cosine
weighting entirely, pick the rank-1 memory, and pre-whiten the residual
bank to amplify the 0.04-nat separation into something the LM can act
on. That is a different method; raw residual replay with soft cosine
gating is closed.

## Artifacts

- `experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_mps_phaseC_pre_block_input/` — full results + analysis.
- `experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_mps_phaseC_mlp_mid/` — full results + analysis.
- `experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_mps_qwen3_4b_phaseA/` (under feat/rsm-fallback-hooks prior smoke) — block_output baseline.

## Reproducibility

```bash
python3 experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --out experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_mps_phaseC_<hook> \
  --device mps --dtype bf16 \
  --hook-point <block_output|pre_block_input|mlp_mid> \
  --seeds 0,1 --eta-grid 0.05,0.10 --theta-grid 0.30,0.50 \
  --bank-size 64 --n-prompts-smoke 24 --n-neutral 16 --phase A

python3 experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/analyze.py \
  --run-dir experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_mps_phaseC_<hook>
```
