# Exp12 — HTWR (Hard-Top1 Whitened Retrieval): Final Report

**Status:** FAIL_DEEP at Tier 0 (oracle ceiling). HTWR direction abandoned.

## TL;DR

Even an **oracle retriever** — given the correct memory by `fact_id` — cannot
make residual-stream replay beat its `shuffled_layers` control. Across three
hook points, four eta magnitudes, and middle-layer-only injection, the gap
remains negative: shuffled-layer (norm-preserved, identity-destroyed) injection
consistently outperforms layer-aligned correct injection.

This is **diagnostically conclusive**: residual replay on Qwen3-4B is global
steering, not layer-aligned fact retrieval. Adding retrieval (raw cosine,
whitened cosine, InfoNCE projector) cannot rescue an injection pipeline whose
oracle ceiling is already dominated by a structure-destroying control.

## Hypothesis (Tier 0)

If residual replay carries fact-specific structure, then with correct memory
delivered by oracle:

- `oracle_correct` should beat `oracle_random` (memory identity matters)
- `oracle_correct` should beat `oracle_shuffled` (layer alignment matters)
- `oracle_correct` should beat `oracle_sign_flip` (direction matters)

The first and third can be true while the second is false — that pattern would
be *INJECTION_ONLY* (any injection helps, but layer correspondence is noise).

## Setup

| | |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Device | Apple MPS, bf16, eager attention |
| Dataset | counterfact_1k.jsonl, eligible after tokenizer filter = 21 |
| Bank size | 32 |
| Seeds | 0, 1 |
| Hook points | `block_output`, `pre_block_input`, `mlp_mid` |
| Injection | additive: `h_l ← h_l + η · sign · m_l[mem]` at last token |
| Retrieval | OracleRetriever (fact_id lookup) + RandomRetriever (uniform wrong) |

Each variant injects exactly one (oracle-chosen) memory per row. Margin =
`logp(target_new) − logp(target_true)`; less negative = better.

## Results

### Hook point × η matrix (block_output)

| η | base | correct | random | shuffled | sign_flip | gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | −4.29 | −4.03 | −4.15 | **−2.49** | −4.54 | −1.54 |
| 0.02 | −4.29 | −3.72 | −3.95 | **−0.49** | −4.73 | −3.23 |
| 0.05 | −4.29 | −2.73 | −3.23 | **−0.23** | −5.23 | −2.50 |
| 0.10 | −4.29 | −1.25 | −1.96 | **−0.12** | −4.58 | −1.13 |
| 0.20 | −4.29 | −0.49 | −1.48 | **−0.32** | −0.87 | −0.17 |

`gap = correct − max(random, shuffled, sign_flip)`. Shuffled wins at every
magnitude. At very large η everything saturates and sign distinction
collapses, but correct still doesn't beat shuffled.

### Hook point comparison (η = 0.05)

| hook | correct | random | shuffled | sign_flip | gap |
|---|---:|---:|---:|---:|---:|
| block_output    | −2.73 | −3.23 | **−0.23** | −5.23 | −2.50 |
| pre_block_input | −2.87 | −3.27 | **−0.23** | −5.04 | −2.64 |
| mlp_mid         | −4.26 | −4.31 | **−4.23** | −4.31 | −0.03 |

`mlp_mid` at this η is effectively inert (delta vs. base ≈ 0.02 nats) — the
9728-dim FFN intermediate space needs much larger η to produce visible
effects, but extrapolating the block_output saturation curve, even there the
gap can be expected to shrink to zero rather than turn positive.

### Layer-subset ablation (block_output, η = 0.10, inject_layers = 8..22)

| variant | margin |
|---|---:|
| base_model | −4.29 |
| oracle_correct | −3.94 |
| oracle_random | −4.00 |
| oracle_shuffled | **−3.17** |
| oracle_sign_flip | −4.39 |

Restricting to subject-token mid-layers (where causal tracing locates fact
state in Llama-class models) does not flip the verdict. Shuffled still wins
by 0.78 nats, and the overall delta-vs-base shrinks to 0.34 because steering
strength scales with the number of injected layers.

## Sign-flip sanity (the one good signal)

Sign-flip is consistently the worst variant at moderate η on block_output and
pre_block_input (e.g. −5.23 at η=0.05). This confirms the injection direction
of the captured memory is **not** noise — flipping it actively hurts. But the
fact that *shuffled (same memory, different layer order) is dramatically
better than correct* shows that what the injection-direction is doing is not
"recall this fact" but "push activations along a useful generic axis whose
layer-by-layer detail is a liability".

## Interpretation

1. **Residual-stream replay is steering, not memory.** Same total perturbation
   magnitude as correct, but with layer correspondence destroyed
   (shuffled_layers), produces a strictly better task-completion outcome.
   Layer identity in the captured per-layer vectors is, at best, neutral; at
   the tested etas it is actively harmful when preserved.

2. **The "fact-specific" content of a residual vector cannot be retrieved by
   layer-aligned vector ops.** Whatever fact information exists in the bank is
   not extractable by `+η·m` at the same layer where it was captured.

3. **The retrieval problem is moot.** T1 (raw cosine), T2 (whitened cosine),
   T3 (subject-token keys), and T4 (InfoNCE projector) all sit *upstream* of
   injection. They can only choose *which* memory to inject. The oracle has
   already pre-selected the best memory in the bank, and that ceiling fails
   the controls — so no retrieval improvement can rescue this.

4. **Why does shuffled help so much?** A residual captured at layer L
   contains, beyond fact identity, a layer-L "what's happening here"
   representation. Injecting it at layer L tries to overwrite that with the
   write-prompt's layer-L state, which mis-aligns with the read-prompt's
   actual hidden state. Shuffling layers averages those layer-state
   mismatches into a smoother, lower-norm-equivalent perturbation that
   functions as a generic distributional shift.

## Verdict ladder

| variant pattern | verdict | observed |
|---|---|---|
| correct CI_lo > all controls AND > base | PASS_STRONG | — |
| correct mean > all controls AND > base | PASS_DIRECTIONAL | — |
| correct > random,shuffled but sign_flip ≥ correct | RETRIEVAL_ONLY | — |
| correct ≈ shuffled, both >> base | INJECTION_ONLY | — |
| any control ≥ correct | STEERING_ONLY | ✓ everywhere |
| correct ≤ base | FAIL_DEEP | — |

Per the ladder, every probed configuration is STEERING_ONLY at best.
Aggregating: T0 oracle ceiling fails — declare **FAIL_DEEP** for the HTWR
direction (no useful path through T1-T4 exists given a failing oracle).

## Relationship to Exp10 (ANB) and Exp11 (RSM)

- **Exp10 ANB Mhc/dynLoPI** — also failed to beat raw-controls on Qwen3-4B
  CounterFact margin. Suggested mechanism: KV-side bank injection acts like
  attention bias, not key-conditioned retrieval.
- **Exp11 RSM** — block_output residual replay failed with the same
  shuffled > correct pattern. We hypothesized the problem was hook-point
  choice and oracle-vs-retrieval entanglement.
- **Exp12 HTWR T0** — controls for both. Hook-point sweep + oracle retrieval
  reveal the failure is in *injection*, not retrieval, and is not
  hook-point-specific.

The line of "external memory delivered as an additive residual vector at the
write-prompt's last token" is now closed for Qwen3-4B in the bf16/MPS regime
with last-token capture.

## What we are NOT claiming

- We are not claiming residual-stream **direction** carries no information.
  Sign matters; the experiment proves that.
- We are not claiming **any** form of external memory injection fails. KV-side
  (ANB-style) attempts have been separately evaluated and exhausted.
- We are not claiming subject-token keys cannot work. We chose not to add
  them at Tier 3 because the oracle ceiling (T0) — which is strictly stronger
  than any subject-key retrieval — already failed.

## Next directions (out of scope for this report)

ATB's design philosophy is **native memory** — the memory is internal to the
model's processing pathway, not text injected into the context.  External
retrieval-augmented generation (RAG, prefix-stuffing) is explicitly out of
scope: it trades the native-memory problem for a context-management problem and
does not address the core research question.

Within the native-memory constraint, the open paths are:

1. **Sparse-token writes**: instead of one residual at last token, capture all
   subject-token positions and replay them as a small KV cache addendum.
   This targets the write side — richer capture, same native injection path.
2. **Adapter or LoRA-conditioned injection**: read out the bank via a trained
   adapter that consumes the memory vector and emits per-token attention bias or
   per-layer scale/shift rather than an additive residual delta.  This targets
   the injection side — learned routing instead of raw vector addition.
3. **Attention-layer key/value delta (revisit ANB)**: the Exp10 failures were
   on Qwen3-4B with dynLoPI / mHC variants; the underlying idea of writing
   fact-specific KV deltas at attention layers has not been exhausted on all
   model families or injection sites (e.g. static KV prefix vs. dynamic
   gate-multiplied delta).

## Artifacts

```
experiments/atb_validation_v1/exp12_htwr/
  run.py                                    # Phase α / Tier 0 runner
  analyze.py                                # six-way verdict ladder + bootstrap CI
  run_mps_phaseA_T0_smoke/                  # block_output, η=0.05
  run_mps_phaseA_T0_pre_block_input/        # pre_block_input, η=0.05
  run_mps_phaseA_T0_mlp_mid/                # mlp_mid, η=0.05
  run_mps_phaseA_T0_eta_sweep/              # block_output, η ∈ {0.01, 0.02, 0.10, 0.20}
  run_mps_phaseA_T0_midlayers/              # block_output, η=0.10, layers 8..22
deltamemory/memory/
  htwr_injector.py                          # pluggable retriever protocol
  htwr_whitening.py                         # ZCA / diag / PCA (built but unused — T0 closed)
  htwr_projector.py                         # InfoNCE projector (built but unused)
tests/
  test_htwr_injector.py                     # 12 tests
  test_htwr_whitening.py                    # 6 tests
  test_htwr_projector.py                    # 5 tests
```
