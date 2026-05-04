# Stage 13B robustness benchmarks — AttentionNative Mneme bank

- model: `google/gemma-4-E2B`
- device/dtype: `mps` / `torch.bfloat16`
- seeds: [0]    alphas: [1.0]
- N facts: 5    paraphrases/fact: 2    decoy targets: 3
- LORO relations: P36

## Gates

| Gate | metric | best | threshold | pass |
|---|---|---|---|---|
| 13B-1_paraphrase | paraphrase_recall_at_1 | 0.200 | 0.7 | FAIL |
| 13B-2_decoy | recall_at_1_vs_K | (diagnostic) | n/a | PASS |
| 13B-3_loro | macro_recall_at_1 | 0.000 | 0.5 | FAIL |

### 13B-1 paraphrase recall@1 (mean across seeds)

- alpha=1.0: 0.200

### 13B-2 decoy curve recall@1 (mean across seeds)

- alpha=1.0: K=0: 0.333  K=5: 0.333

### 13B-3 LORO macro recall@1

- alpha=1.0: 0.000

Per-split (best alpha):
- P36: recall@1=0.000  (n_hold=75)
