# Stage 11 Summary (paired bootstrap 95% CI)

## 11A — paraphrase-augmented InfoNCE encoder

| encoder | held-out paraphrase recall@1 | 95% CI | n | gate G11A ≥ 0.85 |
|---|---:|---|---:|---|
| `multilayer` | 0.138 | [0.134, 0.141] | 3 | ❌ FAIL |
| `prompt_hidden` | 0.053 | [0.049, 0.058] | 3 | ❌ FAIL |

## 11A — decoy ×1000 (regression check on G10B)

| encoder | top1 | 95% CI | n | gate ≥ 0.80 |
|---|---:|---|---:|---|
| `multilayer` | 1.000 | [1.000, 1.000] | 3 | ✅ |
| `prompt_hidden` | 1.000 | [1.000, 1.000] | 3 | ✅ |

## 11A — value ablation (regression check on G10D)

| ablation | top1 | 95% CI | n | gate ≤ 0.10 |
|---|---:|---|---:|---|
| random | 0.000 | [0.000, 0.000] | 6 | ✅ |
| shuffled | 0.009 | [0.007, 0.011] | 6 | ✅ |

## 11B — train-time LORO + adversary

| relation | bind top1 | 95% CI | n | gate ≥ 0.50 |
|---|---:|---|---:|---|
| P101 | 0.000 | [0.000, 0.000] | 3 | ❌ |
| P19 | 0.100 | [0.100, 0.100] | 3 | ❌ |
| P36 | 0.095 | [0.095, 0.095] | 3 | ❌ |
| P39 | 0.000 | [0.000, 0.000] | 3 | ❌ |
| P641 | 0.035 | [0.000, 0.105] | 3 | ❌ |
| P937 | 0.417 | [0.375, 0.438] | 3 | ❌ |
| **overall** | **0.108** | [0.046, 0.178] | 18 | ❌ FAIL |

## 11D — Conversational benchmarks

### D1 multi-turn ConvQA (recall vs filler turns)

| k filler turns | recall@1 | 95% CI | no-leakage |
|---:|---:|---|---:|
| 1 | 1.000 | [1.000, 1.000] | 1.000 |
| 3 | 1.000 | [1.000, 1.000] | 1.000 |
| 5 | 1.000 | [1.000, 1.000] | 1.000 |
| 10 | 1.000 ✅ | [1.000, 1.000] | 1.000 |

### D2 chat-as-write-API vs RAG

| method | top1 | 95% CI |
|---|---:|---|
| DM | 0.967 | [0.950, 1.000] |
| RAG | 0.275 | [0.175, 0.375] |
| DM − RAG | 0.692 | [0.625, 0.775] |

### D3 prompt-injection / poisoning

| metric | value | 95% CI | gate |
|---|---:|---|---|
| overwrite_rate | 0.000 | [0.000, 0.000] | ≤ 0.05 ✅ |
| benign_accept | 0.000 | [0.000, 0.000] | ≥ 0.90 ❌ |
| original_recall | 0.983 | [0.950, 1.000] | ≥ 0.95 ❌ |

## 11E — bit-exact reproduction

- `stage11E_bitexact_run1`: `21035ec0435ef301…`
- `stage11E_bitexact_run2`: `21035ec0435ef301…`

**bit-exact match across runs: ✅ YES**