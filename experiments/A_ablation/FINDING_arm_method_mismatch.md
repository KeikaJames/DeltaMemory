# A 消融矩阵架构问题 (FINDING, 2026-05-06)

## TL;DR

A.2 矩阵 wiring 已 8/8 完成,**但 GB10 Qwen3-4B 旗舰验证跑(160 cells)
显示:除 A5 外,A1/A2/A3/A4/A6/A7 在 method=caa 路径上输出 NLL 与 control
逐位相等**。矩阵不是失效,是**arm 设计与 hot path 错配**。

## 证据

run: `runs/A_ablation_smoke_qwen3/cells.jsonl` on GB10
(Qwen3-4B-Instruct-2507, CUDA bf16, 10 prompts × 8 arms × 2 alphas)

α=1 mean nll over 10 prompts (counterfact_60 prefix):

| arm | mean nll | drift vs control |
|---|---:|---:|
| control | 10.8547 | 0 |
| A1 (post-RoPE K) | 10.8547 | **0.0000** |
| A2 (bank ablation) | 10.8547 | **0.0000** |
| A3 (profiler η_σ=1) | 10.8547 | **0.0000** |
| A4 (SCAR no-M⊥) | 10.8547 | **0.0000** |
| A5 (random steering) | 11.9864 | +1.1317 ✅ |
| A6 (LOPI θ=0) | 10.8547 | **0.0000** |
| A7 (no α-shield) | 10.8547 | **0.0000** |

α=0 redline: 全 7 arm 都 = 7.1177 (control) — 因为 method=caa 在 α=0
时 `hidden + 0·s = hidden` 数学恒等。**这不是 shield 在工作,是乘以 0**。

## 根因 (per arm)

| arm | 设计目标 | 在 method=caa 上的实际效果 |
|---|---|---|
| A1 | 强制 post-RoPE K capture (AttnNativePatcher) | CAA path 不创建 AttnNativePatcher, no-op |
| A2 | bank 消融 (no-bank read) | CAA path 不读 bank, no-op |
| A3 | profiler η_σ=1 (强制广播) | profiler 不在 CAA hot path, no-op |
| A4 | SCAR 移除 M⊥ projection | method=caa 不调 SCAR, no-op |
| A5 | 随机 steering vector | **直接改 inj_ctx.steering_vector, 生效** ✅ |
| A6 | LOPI θ=0 | `CAAConfig(use_lopi_gate=False)` 显式关闭 LOPI, no-op |
| A7 | 移除 α=0 shield | **数值等同**: 移除 `if α<1e-12: return` 不改变 `hidden + 0·s == hidden` |

## 矩阵作为消融工具,实际只能切到对应 method:

- A1, A2: 需要 `--method` 走 bank-attn 路径(eg. attn_native_bank read)
- A3: 需要 `--method` 走 LOPI default(profiler 是 LOPI 输入)
- A4: 需要 `--method scar`
- A5: ✅ method=caa 即可
- A6: 需要 `--method lopi_default`
- A7: 数值上需重设计;当前实现等同 control

## 修法选项

**选项 A — 多 method 矩阵**(推荐,最少代码改动):

```
# CAA arm: 只跑 control + A5 + A7-redesigned
python3 experiments/A_ablation/run.py --method caa --arms control A5 A7

# SCAR arm: 跑 control + A4
python3 experiments/A_ablation/run.py --method scar --arms control A4

# LOPI arm: 跑 control + A3 + A6
python3 experiments/A_ablation/run.py --method lopi_default --arms control A3 A6

# Bank-attn arm: 跑 control + A1 + A2
python3 experiments/A_ablation/run.py --method attn_native --arms control A1 A2
```

需要在 run.py dispatch 里增加 `attn_native` method 或拼装 evaluate_cell。

**选项 B — A7 重新设计**

把"移除 shield"改成"shield 用错误判据"(eg. 用 hidden norm 而非 alpha)
让它在 α=1 也能扰动。

**选项 C — 接受现状**

矩阵 ≠ "compare 8 arms head-to-head on one method" 而是 "factor out
each mechanism on its own primary path"。在 PREREG 里说清楚每个 arm 的
test-vehicle method,分组报告。

## 建议

走选项 C(认知诚实)+ 部分 B(A7 重设计为有数值偏离的 ablation)。
PREREG 已经隐含此分组(eg. A.4 显式 "SCAR 移除 M⊥"),但 A.2 dispatch 把
它们都默认套到 method=caa 上,造成误导。

## 影响

A.3 verdict 不能由此次 GB10 跑产出 — 需要分 method 重跑。但 A.2 wiring
本身没错(单元测试都过),问题在矩阵执行层的 method 选择。

## Next

1. ✅ 把这个 finding commit
2. 改 A.2 PREREG 显式声明每个 arm 的 test-vehicle method
3. 改 dispatch 让 `--arms` 自动选 method,或新增 `--arm-method-map`
4. 重设计 A7(不靠 α=0 路径,改在 α=1 path 上注入扰动)
