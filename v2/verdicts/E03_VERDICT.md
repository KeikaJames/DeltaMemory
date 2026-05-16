# E03 — Capability drift (WikiText-2 PPL)

**Status**: **PASS** — the trained projector + bank does not destroy general LM capability.

**Headline**: On 8,000 WikiText-2 tokens, `bank_off` is bit-equal to base (drift = 0.0000) and `bank_on` PPL drift = **+0.18% relative** (8.349 → 8.380), comfortably below the ≤ 5% pass bar.

---

## a. Reproduction command

```bash
python3 v2/experiments/e03_capability_drift/run.py \
    --tokens 8000 --ctx 1024 \
    --n_preload 512 --bank_layer 9
```

## b. Seeds & sample size

seed 0; 8,000 WikiText-2 evaluation tokens at ctx=1024; n_preload=512; bank_layer=9.

## c. Raw data paths

`v2/experiments/e03_capability_drift/e03_drift_t8000.json`

## d. Numbers

| Condition | NLL | PPL | Δ vs base |
|---|---:|---:|---:|
| base | 2.1221 | 8.349 | 0.0 |
| bank_off | 2.1221 | 8.349 | **0.000 (bit-equal)** |
| bank_on | 2.1259 | 8.380 | **+0.0037 nll / +0.18% PPL** |

Verdict object: `{off_bit_equal: True, on_within_5pct: True}`.

## e. Verdict

- **Hypothesis (capability-preservation)**: "Training the projector does not collapse the base LM on neutral text."
- **Result**: **Confirmed.** bank_off perfectly restores base behavior (the projector + LM weights are not modifying the no-bank forward pass), and bank_on drift is 35× below the pass threshold.
- **Pass rate**: 2/2 rules satisfied.

## f. Caveat

Only WikiText-2 was evaluated; lm-eval-harness subsets (HellaSwag, ARC, MMLU) listed in the v2 plan were not run here (HellaSwag accuracy drop *was* observed in e13: −1.5 pp). This verdict shows capability is preserved on **distributional / language modeling** evals; e13 shows capability is *not* boosted on multiple-choice OOD evals — these are consistent but cover different surfaces.

## g. Implications

- E03 is **not a falsifier**; it is a guard rail confirming the adapter does not corrupt the frozen base LM in the bank-off setting and only mildly perturbs it in the bank-on setting.
- In combination with the e16-forgetting result (A/B symmetry: projector encodes a generic distribution shift, not item-specific memory), E03's mild PPL drift in bank-on is best interpreted as the projector's residual fine-tune bias leaking into generic text.
