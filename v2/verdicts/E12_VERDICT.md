# E12 — Long-term / Short-term bank coexistence

**Status**: **PARTIAL** — LT-only works, ST-only works, but they interfere when mixed (the "no-interference" rule fails).

**Headline**: With a 256-entry long-term (LT) bank preloaded and a 64-entry short-term (ST) bank written at inference, both channels independently improve NLL, but their joint use does **not** preserve either channel's standalone performance — LT items lose accuracy under ST presence and/or vice versa.

---

## a. Reproduction command

```bash
python3 v2/experiments/e12_LT_ST_coexist/run.py --seed 0 \
    --n_LT 256 --n_ST 64 \
    --bank_layer 9 --rank 64 --steps 200 \
    --n_train 120 --n_eval 120
```

## b. Seeds & sample size

seed 0; n_LT=256, n_ST=64; n_train=120, n_test=120; bank_layer=9; rank=64; steps=200.

## c. Raw data paths

`v2/experiments/e12_LT_ST_coexist/e12_seed0.json`

## d. Numbers

Verdict object:

```json
{
  "pass_LT_only": true,
  "pass_ST_works": true,
  "pass_no_interference": false,
  "overall": false
}
```

Each cell (`LT_only`, `ST_only`, `LT_ST_mix_LT_items`, `LT_ST_mix_ST_items`) contains base/post NLL; on mixing, both LT-targeted and ST-targeted evals degrade vs their pure-channel runs.

## e. Verdict

- **Hypothesis ("LT and ST coexist non-destructively")**: Refuted on the no-interference rule.
- **Pass rate**: 2/3 (LT-only OK, ST-works OK, no-interference FAIL).
- This is consistent with the adapter interpretation: a single rank-64 projector cannot simultaneously fit two distinct distribution shifts (LT-conditioned and ST-conditioned) without trading capacity between them.

## f. Caveat

- Single seed; single rank (64).
- A larger projector (rank 128/256) or per-channel projectors (separate P_LT and P_ST) was not explored. The attention-side latent bank-inspired thesis hypothesized one shared bank — that exact configuration is what fails here. A *per-channel* projector design could plausibly pass.

## g. Implications

- The "long+short memory in one AttentionBank" component of the ALB thesis does not work as currently architected at rank-64.
- If revisited, the architectural fix is per-channel projectors (or per-channel sub-banks with disjoint gating). That moves the design further from the attention-side latent bank single-store metaphor and closer to a routed multi-adapter system.
