# DeltaMemory v3 Intervention Demo — `Qwen/Qwen3-4B-Instruct-2507`

- adapter: `qwen3`  | layers: 36  | device: `cuda`  | dtype: `bfloat16`  | alpha: 0.05  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| ff1_paris_mayor_napoleon | ` Napoleon` | -12.005 | -2.776 | -11.240 | +0.764 | -8.465 |
| ff2_eiffel_arch_picasso | ` Pablo` | -21.093 | -3.844 | -20.849 | +0.244 | -17.006 |
| ff3_mona_lisa_van_gogh | ` Vincent` | -10.253 | -7.071 | -9.401 | +0.852 | -2.329 |
| ff4_relativity_newton | ` Isaac` | -7.034 | -10.368 | -5.988 | +1.047 | +4.381 |
| ff5_python_lovelace | ` Ada` | -13.770 | -8.555 | -12.324 | +1.446 | -3.769 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### ff1_paris_mayor_napoleon — Q about *the mayor of Paris* (target = ` Napoleon`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` As` | 0.5165 |
| 2 | ` The` | 0.3550 |
| 3 | ` Paris` | 0.1017 |
| 4 | ` There` | 0.0042 |
| 5 | ` as` | 0.0031 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8601 |
| 2 | ` Napoleon` | 0.0623 |
| 3 | ` *` | 0.0428 |
| 4 | ` It` | 0.0123 |
| 5 | ` Let` | 0.0058 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.5405 |
| 2 | ` As` | 0.3715 |
| 3 | ` Paris` | 0.0286 |
| 4 | ` the` | 0.0072 |
| 5 | ` In` | 0.0041 |

### ff2_eiffel_arch_picasso — Q about *the architect of the Eiffel Tower* (target = ` Pablo`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.9316 |
| 2 | ` Gust` | 0.0526 |
| 3 | ` the` | 0.0055 |
| 4 | ` G` | 0.0012 |
| 5 | ` Charles` | 0.0010 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8036 |
| 2 | ` What` | 0.0960 |
| 3 | ` *` | 0.0275 |
| 4 | ` Pablo` | 0.0214 |
| 5 | ` Let` | 0.0167 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8809 |
| 2 | ` Gust` | 0.0928 |
| 3 | ` the` | 0.0052 |
| 4 | ` G` | 0.0026 |
| 5 | ` A` | 0.0025 |

### ff3_mona_lisa_van_gogh — Q about *the painter of the Mona Lisa* (target = ` Vincent`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.7293 |
| 2 | ` The` | 0.2683 |
| 3 | ` the` | 0.0007 |
| 4 | ` Leon` | 0.0005 |
| 5 | ` Le` | 0.0003 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8746 |
| 2 | ` *` | 0.0718 |
| 3 | ` It` | 0.0182 |
| 4 | ` Step` | 0.0110 |
| 5 | ` What` | 0.0036 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.5552 |
| 2 | ` The` | 0.4324 |
| 3 | ` A` | 0.0026 |
| 4 | ` the` | 0.0020 |
| 5 | ` Leon` | 0.0011 |

### ff4_relativity_newton — Q about *the discoverer of general relativity* (target = ` Isaac`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.7524 |
| 2 | ` General` | 0.1018 |
| 3 | ` Einstein` | 0.0481 |
| 4 | ` general` | 0.0375 |
| 5 | ` The` | 0.0331 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6918 |
| 2 | ` *` | 0.1982 |
| 3 | ` It` | 0.0344 |
| 4 | ` \n` | 0.0077 |
| 5 | ` (` | 0.0064 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.6956 |
| 2 | ` General` | 0.1067 |
| 3 | ` Einstein` | 0.0647 |
| 4 | ` The` | 0.0369 |
| 5 | ` A` | 0.0197 |

### ff5_python_lovelace — Q about *the creator of the Python language* (target = ` Ada`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6949 |
| 2 | ` Python` | 0.1757 |
| 3 | ` Guid` | 0.1208 |
| 4 | ` **` | 0.0016 |
| 5 | ` In` | 0.0013 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8893 |
| 2 | ` *` | 0.0237 |
| 3 | ` A` | 0.0127 |
| 4 | ` Step` | 0.0112 |
| 5 | ` Correct` | 0.0105 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6793 |
| 2 | ` Guid` | 0.1946 |
| 3 | ` Python` | 0.1042 |
| 4 | ` **` | 0.0071 |
| 5 | ` In` | 0.0019 |

## Aggregate
- mean Δ logprob v3 − B0 = **+0.871**
- mean Δ logprob B1 − B0 = **+6.308**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-5.438**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model Qwen/Qwen3-4B-Instruct-2507 --device cuda --dtype bfloat16 --alpha 0.05 --false-facts --out-dir transcripts/v31_intervention/qwen3-4b-gb10-FALSE
```