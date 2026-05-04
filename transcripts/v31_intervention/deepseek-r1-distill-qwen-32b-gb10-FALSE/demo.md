# Mneme v3 Intervention Demo — `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

- adapter: `llama`  | layers: 64  | device: `cuda`  | dtype: `bfloat16`  | alpha: 0.05  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| ff1_paris_mayor_napoleon | ` Napoleon` | -14.946 | -0.734 | -14.885 | +0.060 | -14.152 |
| ff2_eiffel_arch_picasso | ` Pablo` | -14.904 | -0.358 | -15.050 | -0.147 | -14.692 |
| ff3_mona_lisa_van_gogh | ` Vincent` | -10.436 | -2.308 | -11.880 | -1.444 | -9.572 |
| ff4_relativity_newton | ` Isaac` | -9.491 | -2.010 | -10.193 | -0.702 | -8.183 |
| ff5_python_lovelace | ` Ada` | -13.867 | -0.432 | -14.595 | -0.728 | -14.163 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### ff1_paris_mayor_napoleon — Q about *the mayor of Paris* (target = ` Napoleon`)

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
| 1 | ` Napoleon` | 0.4800 |
| 2 | ` The` | 0.4236 |
| 3 | ` [` | 0.0271 |
| 4 | ` \n\n` | 0.0198 |
| 5 | ` I` | 0.0047 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8204 |
| 2 | ` Anne` | 0.0594 |
| 3 | ` \n\n` | 0.0193 |
| 4 | ` As` | 0.0091 |
| 5 | ` I` | 0.0080 |

### ff2_eiffel_arch_picasso — Q about *the architect of the Eiffel Tower* (target = ` Pablo`)

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
| 1 | ` Pablo` | 0.6990 |
| 2 | ` The` | 0.2003 |
| 3 | ` I` | 0.0106 |
| 4 | ` \n\n` | 0.0088 |
| 5 | ` [` | 0.0083 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8657 |
| 2 | ` Gust` | 0.0431 |
| 3 | ` \n\n` | 0.0123 |
| 4 | ` It` | 0.0085 |
| 5 | ` A` | 0.0080 |

### ff3_mona_lisa_van_gogh — Q about *the painter of the Mona Lisa* (target = ` Vincent`)

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
| 1 | ` The` | 0.6090 |
| 2 | ` Vincent` | 0.0994 |
| 3 | ` Leonardo` | 0.0500 |
| 4 | ` [` | 0.0323 |
| 5 | ` \n\n` | 0.0251 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6033 |
| 2 | ` Leonardo` | 0.2085 |
| 3 | ` It` | 0.0220 |
| 4 | ` A` | 0.0206 |
| 5 | ` \n\n` | 0.0182 |

### ff4_relativity_newton — Q about *the discoverer of general relativity* (target = ` Isaac`)

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
| 1 | ` General` | 0.5297 |
| 2 | ` Isaac` | 0.1339 |
| 3 | ` [` | 0.0812 |
| 4 | ` Albert` | 0.0493 |
| 5 | ` The` | 0.0384 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.5158 |
| 2 | ` General` | 0.2761 |
| 3 | ` Einstein` | 0.0579 |
| 4 | ` The` | 0.0423 |
| 5 | ` \n\n` | 0.0273 |

### ff5_python_lovelace — Q about *the creator of the Python language* (target = ` Ada`)

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
| 1 | ` Ada` | 0.6494 |
| 2 | ` Python` | 0.0936 |
| 3 | ` Guid` | 0.0567 |
| 4 | ` The` | 0.0470 |
| 5 | ` [` | 0.0415 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guid` | 0.3561 |
| 2 | ` The` | 0.2774 |
| 3 | ` Python` | 0.2160 |
| 4 | ` [` | 0.0228 |
| 5 | ` \n` | 0.0189 |

## Aggregate
- mean Δ logprob v3 − B0 = **-0.592**
- mean Δ logprob B1 − B0 = **+11.560**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-12.152**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --device cuda --dtype bfloat16 --alpha 0.05 --false-facts --out-dir transcripts/v31_intervention/deepseek-r1-distill-qwen-32b-gb10-FALSE
```