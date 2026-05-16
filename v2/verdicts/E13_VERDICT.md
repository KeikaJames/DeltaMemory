# E13 — Multi-task capability (transfer)

**Status**: The "memory transfers across tasks" claim is **falsified** by E13.
**Headline**: A projector trained on factual-completion (MEMIT-style) and evaluated on three out-of-distribution tasks shows **zero positive transfer**: WikiText-2 PPL *increases* by 0.4, Lambada accuracy is exactly unchanged (Δacc=0.000), HellaSwag accuracy *drops* by 1.5 pp. The adapter is task-specific.

(Partial result. GSM8K eval was attempted but hung in uninterruptible-sleep state for 50+ min on MPS and had to be killed; not included in headline numbers. The three completed tasks suffice to falsify the transfer claim.)

---

## a. Reproduction command

```bash
python3 v2/experiments/e13_multi_task_capability/run.py --seed 0 \
    --variant multitask_triple_partial \
    --bank_layer 9 --rank 64 --steps 200 \
    --n_train 120 --n_preload 512 \
    --tasks wikitext2,lambada,hellaswag
```

## b. Seeds & sample size

seed 0; n_train=120 factual-completion examples; n_preload=512; bank_layer=9; rank=64; steps=200.

Eval sizes:
- WikiText-2: 3,996 tokens
- Lambada: 227 examples (correct/total format)
- HellaSwag: 200 examples
- GSM8K: skipped (MPS-generation hang)

## c. Raw data paths

`v2/experiments/e13_multi_task_capability/e13_multitask_partial_seed0.json`

## d. Numbers

| Task | metric | base | bank_off | bank_on | Δ (bank_on − base) |
|---|---|---:|---:|---:|---:|
| WikiText-2 | NLL | 2.0644 | 2.0644 | 2.1144 | **+0.050** (PPL 7.88 → 8.29) |
| WikiText-2 | PPL | 7.880 | 7.880 | 8.285 | **+0.405** |
| Lambada | accuracy | 0.5727 | 0.5727 | 0.5727 | **0.000** |
| HellaSwag | accuracy | 0.6500 | 0.6500 | 0.6350 | **−0.015** |
| GSM8K | EM | (incomplete — MPS generation hang) |

`bank_off = base` perfectly across tasks: removing the bank at eval after training restores the model exactly (with the trained projector in place). `bank_on` differs only on WikiText-2 (slightly worse) and HellaSwag (slightly worse).

## e. Verdict

- **Hypothesis**: "If the bank+projector encodes useful general knowledge, training on one task should benefit (or at least not harm) related tasks."
- **Result**: **Refuted.** On three independent OOD tasks, the trained mechanism produces effects in {−0.015 pp acc, 0.000 pp acc, +0.4 PPL}. None of these are positive transfer; two are slightly negative.
- **Pass rate**: 0/3 on the rule "≥ 3 of 4 tasks show Δ ≥ +2 pp acc or ΔNLL ≤ −0.1."
- **Falsifier #3 in V2_FINAL_VERDICT §1.Overall Stance.**

## f. Caveat

- **GSM8K incomplete**: The model.generate()-based EM eval on Qwen3-4B MPS entered uninterruptible-sleep after 50 min and was killed. A teacher-forced NLL on GSM8K reference solutions would be the safer eval strategy if GSM8K-specific number is desired.
- **Single seed**: Replications across seeds {1, 2} were not run; if seed-0 transfer is anomalously negative, the picture might shift slightly. However the magnitude observed (−0.015 to +0.05 nat) is far inside any plausible seed-noise floor.

## g. Implications

- The projector is a *task-specific* adapter — it learns the specific distribution of (template, target) pairs it was trained on and gives essentially no boost to even mildly OOD eval distributions.
- This combined with e17 (template-conditional content blindness) and e16-forgetting (A/B symmetry) paints a coherent picture: the projector is a low-rank fine-tune that fits its training distribution and applies a similar "distributional shift" to anything that closely matches that distribution. It is not memory and does not transfer.
