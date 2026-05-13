# X.2 — 矛盾事实预注册 (X2.v1)

## 动机

回应用户疑问①的反面:bank 不是被 N→∞ "埋葬"(X.1 已证伪),
但当 bank 内部存在**对立事实**时,谁胜出?写入顺序是否敏感?
LRU 容量+读取节奏能否仲裁?

X.1 测的是"bank 大就稀释",X.2 测的是"bank 内信息冲突时的仲裁逻辑"。
两者互补,共同覆盖 bank attention 长期累积的失败模态。

## 数据集

复用 `experiments/X1_bank_scaling/facts.jsonl` 的 target (Mount Everest /
continent / Asia)。在此之上构造**对立对**:

- `target_A`: write_prompt = "Fact: Mount Everest is on the continent of **Antarctica**."
- `target_B`: write_prompt = "Fact: Mount Everest is on the continent of **Africa**."
- `target_canon`: "Asia" (基线锚点)

distractor 流复用 `X1_bank_scaling/distractors.jsonl`。

## 单元 (cells)

| 维度 | 取值 |
|---|---|
| 模型 | `google/gemma-3-1b-it` (旗舰下限,MPS bf16) |
| 写入顺序 | `A_first` (写A→B), `B_first` (写B→A) |
| 中间distractor数 N | {0, 100, 1000} (between A 和 B 之间) |
| 容量 cap | {0=无界, 64, 256} |
| 驱逐策略 | {lru, fifo} (cap=0 仅 lru) |
| α | {0.0=红线, 1.0} |
| seed | {0, 1, 2} |

每 cell 度量:
- `log_margin_A_vs_B = score(Antarctica) - score(Africa)`  
- `log_margin_winner_vs_canon`
- `score_A`, `score_B`, `score_canon`
- `target_A_resident`, `target_B_resident`, `bank_size`
- `read_latency_ms`

## 假设

**H_X2.0 (α=0 redline)**: 当 α=0,任何写入顺序都不应该改变 logits。  
**判据**: 所有 α=0 cell 的 `log_margin_A_vs_B` 在 abstol=1e-3 内一致。  
**违反则整轮无效**(说明 α-shield 失效)。

**H_X2.1 (后写覆盖 / recency wins)**: 在 cap=0 (无界)、N=0 (无干扰) 下,
后写入的事实应更强。  
**判据**: `A_first` cell 的 `log_margin_A_vs_B < 0` (B 胜),  
`B_first` cell 的 `log_margin_A_vs_B > 0` (A 胜)。  
**支持需**: ≥80% (N=0, cap=0, α=1) cells 满足"后写胜"。

**H_X2.2 (LRU 距离敏感)**: 当 cap=64, N=1000 时,先写入的 target 在
distractor 流中可能被 LRU 驱逐(若无中间读取)。  
**判据**: `A_first` (A 在很多 distractor 之前) 的 `target_A_resident` 
应低于 `B_first` (A 在 B 之后才写入,等于刚写入)。  
**支持需**: `mean(A_resident | A_first) < mean(A_resident | B_first) - 0.2`。

**H_X2.3 (FIFO 顺序刚性)**: FIFO 下,先写入的 target 在 N>cap 时一定被驱逐,
不论中间是否被读取。  
**判据**: `A_first, FIFO, N>cap`: `target_A_resident == False` 在所有 cell。

## 运行

```
python3 experiments/X2_contradictory/run.py \
  --out runs/X2_full_v1 --device mps --dtype bf16
```

冒烟:
```
python3 experiments/X2_contradictory/run.py --smoke --out runs/X2_smoke
```

## 完整性

- `env.json`: prereg_version, dataset_sha1, device, dtype, cli_argv, model
- `cells.jsonl`: append-only,SHA-stable cell_id
- `summary.json`: H_X2.0/1/2/3 verdicts + per-condition aggregates
- `REPORT.md`: 解读 + 可发表结论
