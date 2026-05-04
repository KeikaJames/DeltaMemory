# Phase R-5.1 — Q3 Adversarial Chat × LOPI (Gemma-4-E2B pilot)

**Date**: 2026-05-04
**Branch / commit**: `stage13-attn-native` post-`8547e074`
**Hardware**: Mac M-series, MPS bf16 (Gemma-4-E2B fits comfortably)
**Runner**: `scripts/run_flagship_q3_adversarial_chat.py --lopi both`
**Facts**: 5 counter-prior pilot facts (`FALSE_FACTS` in `run_intervention_demo`)
**Decoding**: greedy, 64 new tokens, shield ON in both arms

## Setup

For each α ∈ {2, 5, 8, 10} we generated three completions per fact:

1. **baseline**  — frozen LLM, no bank, no injection;
2. **shield ON, LOPI=off** — shield-only column-cap;
3. **shield ON, LOPI=on**  — shield + LOPI v3.4 (gauss + γ, no orthogonal).

Implant labels are computed by `evaluate_implant`:
* `accurate_implant` — false object literal appears in generated text;
* `partial_implant` — at least one object token in the first sentence;
* `not_implanted` — neither.

## Results

| α | baseline FP | LOPI off (acc / partial) | LOPI on (acc / partial) | Δ-acc | Δ-partial |
|---:|:---:|:---:|:---:|:---:|:---:|
| 2  | 0/5 | 0/5 / 0/5 | 0/5 / 0/5 | +0pp | +0pp |
| 5  | 0/5 | 0/5 / 1/5 | 0/5 / 1/5 | +0pp | +0pp |
| 8  | 0/5 | 0/5 / 0/5 | 0/5 / 1/5 | +0pp | **+20pp** |
| 10 | 0/5 | 0/5 / 0/5 | 0/5 / 1/5 | +0pp | **+20pp** |

Where LOPI elevates a fact above the partial threshold the per-fact pair is:

| α | fact_id | base | LOPI off | LOPI on |
|---:|---|---|---|---|
| 8  | `ff1_paris_mayor_napoleon` | not_implanted | not_implanted | **partial_implant** |
| 10 | `ff1_paris_mayor_napoleon` | not_implanted | not_implanted | **partial_implant** |

`ff1_paris_mayor_napoleon` is the easiest fact in the pilot set (Napoleon is a
high-frequency token); LOPI is the only configuration that surfaces a
fragment of it past the shield + greedy decoder.

## Verdict against L5 (preregistered)

**L5 PASS criterion**: `accurate_implant_rate(LOPI on) − accurate_implant_rate(LOPI off) ≥ +20pp` on at least one model.

- Gemma-4-E2B accurate rate: **0/5 in every (α, LOPI) cell**.
- L5 strict (accurate-rate uplift): **NOT MET**.
- L5-relaxed (any-implant uplift, partial allowed): **MET at α∈{8, 10}**
  (+20pp partial-rate, 0 → 1/5 each, exclusively under LOPI).

## Interpretation

R-4 already showed Gemma-4-E2B α=10 lift = +4.96 nats under shield+LOPI (best
across 24 cells). The R-5.1 finding is that **+4.96 nats of NLL lift on a single
target token does not translate into greedy-decoder accurate implants on this
5-fact pilot set**. The bank does flip the *partial* implant of one fact under
LOPI alone, which is a strict improvement over shield-only (0/5 → 1/5 partial),
but the accurate-implant threshold is not crossed.

This is consistent with the preregistered Q3 finding (`flagship_v32/Q3` at
α=1.0 already gave 0/5 accurate). The shield + LOPI combination extends the
safe α-window to α=10 but the greedy-decode emission still requires α to push
the *target token* into rank-1, not just rank-2.

## Honest limitations

1. **Greedy decode is a binary discriminator on rank-1** — partial-implant
   margin gains are not accurate-implant margin gains. A sampling-based or
   logit-bias-based eval might pick up the +4.96 nats lift more sensitively.
2. **5-fact pilot is under-powered** — partial-rate uplift of 1/5 (20pp) is
   not statistically distinguishable from noise. R-5.2 (60-fact extension)
   is needed to call any L5 verdict with confidence.
3. **Single-seed greedy** — no seed variance. Sampling at temperature ≥ 0.5
   would let LOPI's logit lift translate to occasional accurate implants.
4. **Qwen / GLM not run** — Q2 + R-4 already established absolute drift
   floors (3.6 / 1.8 nats) caused by v_norm absence; chat implant on those
   architectures will collapse first into incoherent text and then into
   target-token emission. Defer to R-5.2 with `--lopi on` only.

## Next steps

* R-5.2: 60-fact LAMA + ConceptNet extension (single Gemma run, α=10, shield+LOPI).
* R-5.3: sampling-based eval (temperature 0.7, n=8 generations / fact).
* R-5.4: Multi-fact bank stress (8 facts simultaneously, ask only one).
* R-6 (parallel): persistent safetensors+FAISS bank store (foundation for R-5.4).

## Reproduction

```bash
for a in 2.0 5.0 8.0 10.0; do
  .venv-mac/bin/python scripts/run_flagship_q3_adversarial_chat.py \
    --model google/gemma-4-E2B --device mps --dtype bfloat16 \
    --alpha $a --facts pilot --lopi both \
    --out reports/cleanroom/lopi_v33/R5_q3
done
```

Artefacts: `reports/cleanroom/lopi_v33/R5_q3/google_gemma-4-E2B/{summary,results}_alpha{2p0,5p0,8p0,10p0}.{json,jsonl}`.
