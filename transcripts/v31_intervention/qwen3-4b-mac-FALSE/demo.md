# Mneme v3 Intervention Demo — `Qwen/Qwen3-4B-Instruct-2507`

- adapter: `qwen3`  | layers: 36  | device: `mps`  | dtype: `bfloat16`  | alpha: 0.05  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| ff1_paris_mayor_napoleon | ` Napoleon` | -11.816 | -2.898 | -11.274 | +0.543 | -8.376 |
| ff2_eiffel_arch_picasso | ` Pablo` | -21.279 | -4.759 | -20.783 | +0.496 | -16.024 |
| ff3_mona_lisa_van_gogh | ` Vincent` | -10.264 | -7.718 | -9.410 | +0.855 | -1.692 |
| ff4_relativity_newton | ` Isaac` | -6.981 | -10.226 | -5.971 | +1.010 | +4.255 |
| ff5_python_lovelace | ` Ada` | -13.833 | -8.412 | -12.238 | +1.595 | -3.826 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### ff1_paris_mayor_napoleon — Q about *the mayor of Paris* (target = ` Napoleon`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` As` | 0.4855 |
| 2 | ` The` | 0.3781 |
| 3 | ` Paris` | 0.1083 |
| 4 | ` There` | 0.0048 |
| 5 | ` I` | 0.0029 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8626 |
| 2 | ` Napoleon` | 0.0551 |
| 3 | ` *` | 0.0429 |
| 4 | ` It` | 0.0158 |
| 5 | ` Let` | 0.0058 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.5394 |
| 2 | ` As` | 0.3707 |
| 3 | ` Paris` | 0.0252 |
| 4 | ` the` | 0.0077 |
| 5 | ` In` | 0.0047 |

### ff2_eiffel_arch_picasso — Q about *the architect of the Eiffel Tower* (target = ` Pablo`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.9375 |
| 2 | ` Gust` | 0.0467 |
| 3 | ` the` | 0.0063 |
| 4 | ` G` | 0.0010 |
| 5 | ` Charles` | 0.0009 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8744 |
| 2 | ` What` | 0.0559 |
| 3 | ` *` | 0.0181 |
| 4 | ` Let` | 0.0141 |
| 5 | ` Pablo` | 0.0086 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8672 |
| 2 | ` Gust` | 0.1036 |
| 3 | ` the` | 0.0066 |
| 4 | ` G` | 0.0029 |
| 5 | ` A` | 0.0028 |

### ff3_mona_lisa_van_gogh — Q about *the painter of the Mona Lisa* (target = ` Vincent`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.6775 |
| 2 | ` The` | 0.3200 |
| 3 | ` the` | 0.0008 |
| 4 | ` Leon` | 0.0005 |
| 5 | ` Le` | 0.0003 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8560 |
| 2 | ` *` | 0.0902 |
| 3 | ` It` | 0.0228 |
| 4 | ` Step` | 0.0108 |
| 5 | ` What` | 0.0026 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.5858 |
| 2 | ` The` | 0.4026 |
| 3 | ` A` | 0.0021 |
| 4 | ` the` | 0.0016 |
| 5 | ` Leon` | 0.0011 |

### ff4_relativity_newton — Q about *the discoverer of general relativity* (target = ` Isaac`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.7456 |
| 2 | ` General` | 0.0891 |
| 3 | ` general` | 0.0540 |
| 4 | ` Einstein` | 0.0540 |
| 5 | ` The` | 0.0308 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6614 |
| 2 | ` *` | 0.2147 |
| 3 | ` It` | 0.0479 |
| 4 | ` A` | 0.0061 |
| 5 | ` *\n` | 0.0061 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.7076 |
| 2 | ` General` | 0.0958 |
| 3 | ` Einstein` | 0.0658 |
| 4 | ` The` | 0.0375 |
| 5 | ` A` | 0.0214 |

### ff5_python_lovelace — Q about *the creator of the Python language* (target = ` Ada`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6947 |
| 2 | ` Python` | 0.1756 |
| 3 | ` Guid` | 0.1207 |
| 4 | ` **` | 0.0018 |
| 5 | ` In` | 0.0013 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.9051 |
| 2 | ` *` | 0.0273 |
| 3 | ` Step` | 0.0101 |
| 4 | ` A` | 0.0083 |
| 5 | ` Correct` | 0.0083 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6533 |
| 2 | ` Guid` | 0.2121 |
| 3 | ` Python` | 0.1135 |
| 4 | ` **` | 0.0068 |
| 5 | ` In` | 0.0018 |

## Aggregate
- mean Δ logprob v3 − B0 = **+0.900**
- mean Δ logprob B1 − B0 = **+6.032**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-5.133**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model Qwen/Qwen3-4B-Instruct-2507 --device mps --dtype bfloat16 --alpha 0.05 --false-facts --out-dir transcripts/v31_intervention/qwen3-4b-mac-FALSE
```