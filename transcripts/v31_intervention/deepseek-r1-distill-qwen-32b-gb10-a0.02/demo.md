# DeltaMemory v3 Intervention Demo — `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

- adapter: `llama`  | layers: 64  | device: `cuda`  | dtype: `bfloat16`  | alpha: 0.02  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| f1_paris_mayor | ` Anne` | -2.665 | -1.148 | -2.965 | -0.301 | -1.817 |
| f2_eiffel_arch | ` Gust` | -2.622 | -1.216 | -2.963 | -0.340 | -1.747 |
| f3_mona_lisa | ` Leonardo` | -1.624 | -1.837 | -1.306 | +0.318 | +0.531 |
| f4_relativity | ` Albert` | -0.491 | -0.551 | -0.375 | +0.116 | +0.176 |
| f5_python_creator | ` Guid` | -1.773 | -1.253 | -1.078 | +0.695 | +0.176 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### f1_paris_mayor — Q about *the mayor of Paris* (target = ` Anne`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8483 |
| 2 | ` Anne` | 0.0696 |
| 3 | ` As` | 0.0155 |
| 4 | ` Ber` | 0.0137 |
| 5 | `<｜end▁of▁sentence｜>` | 0.0083 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.5229 |
| 2 | ` Anne` | 0.3172 |
| 3 | ` [` | 0.0967 |
| 4 | ` \n\n` | 0.0245 |
| 5 | ` I` | 0.0035 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8582 |
| 2 | ` Anne` | 0.0515 |
| 3 | ` \n\n` | 0.0190 |
| 4 | `<｜end▁of▁sentence｜>` | 0.0079 |
| 5 | ` I` | 0.0074 |

### f2_eiffel_arch — Q about *the architect of the Eiffel Tower* (target = ` Gust`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8849 |
| 2 | ` Gust` | 0.0726 |
| 3 | ` \n\n` | 0.0056 |
| 4 | ` Alexandre` | 0.0053 |
| 5 | ` I` | 0.0044 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.5205 |
| 2 | ` Gust` | 0.2966 |
| 3 | ` It` | 0.0377 |
| 4 | ` [` | 0.0377 |
| 5 | ` \n\n` | 0.0229 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8606 |
| 2 | ` Gust` | 0.0517 |
| 3 | ` It` | 0.0115 |
| 4 | ` Alexandre` | 0.0102 |
| 5 | ` A` | 0.0090 |

### f3_mona_lisa — Q about *the painter of the Mona Lisa* (target = ` Leonardo`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.7326 |
| 2 | ` Leonardo` | 0.1972 |
| 3 | ` I` | 0.0162 |
| 4 | ` A` | 0.0060 |
| 5 | ` \n\n` | 0.0046 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.4609 |
| 2 | ` [` | 0.1805 |
| 3 | ` Leonardo` | 0.1593 |
| 4 | ` \n\n` | 0.0456 |
| 5 | ` It` | 0.0115 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.5389 |
| 2 | ` Leonardo` | 0.2710 |
| 3 | ` A` | 0.0222 |
| 4 | ` \n\n` | 0.0184 |
| 5 | ` I` | 0.0173 |

### f4_relativity — Q about *the discoverer of general relativity* (target = ` Albert`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.6120 |
| 2 | ` General` | 0.2891 |
| 3 | ` Einstein` | 0.0502 |
| 4 | ` The` | 0.0163 |
| 5 | ` \n\n` | 0.0047 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.5761 |
| 2 | ` General` | 0.2402 |
| 3 | ` [` | 0.0884 |
| 4 | ` The` | 0.0223 |
| 5 | ` \n\n` | 0.0174 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.6870 |
| 2 | ` Einstein` | 0.0990 |
| 3 | ` General` | 0.0873 |
| 4 | ` The` | 0.0302 |
| 5 | ` It` | 0.0104 |

### f5_python_creator — Q about *the creator of the Python language* (target = ` Guid`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.4617 |
| 2 | ` Python` | 0.2471 |
| 3 | ` Guid` | 0.1699 |
| 4 | ` \n\n` | 0.0625 |
| 5 | ` I` | 0.0179 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guid` | 0.2855 |
| 2 | ` The` | 0.2224 |
| 3 | ` [` | 0.2224 |
| 4 | ` \n\n` | 0.0818 |
| 5 | ` Python` | 0.0562 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guid` | 0.3404 |
| 2 | ` Python` | 0.2651 |
| 3 | ` The` | 0.2340 |
| 4 | ` I` | 0.0247 |
| 5 | ` [` | 0.0204 |

## Aggregate
- mean Δ logprob v3 − B0 = **+0.097**
- mean Δ logprob B1 − B0 = **+0.634**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-0.536**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --device cuda --dtype bfloat16 --alpha 0.02
```