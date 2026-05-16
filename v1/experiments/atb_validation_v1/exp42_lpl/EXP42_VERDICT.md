# Exp42 — Latent Pause Loop + AttentionBank — VERDICT

## TL;DR

LPL 架构在 Qwen3-4B-Instruct-2507 上**部分成立**，关键发现按重要性排序：

1. **【主成果】静态记忆桥的"K-projector 复活"** — 把 Exp35b 的 10K MEMIT b-vector
   作为 LPL AttentionBank 预填充，再训一个 rank-64 的 bank-side K/V projector
   （420K 可训练参数，整 4B base 冻结），在 held-out test-split 事实回忆上
   NLL **12.13 → 6.30 (Δ=−5.83)**；同体量随机 bank + 同一 projector 仅 Δ=−0.02。
   - 这复活了 v1 AttnNativeBank 公式中**最关键也最被静态化忽视的环节**：
     可学习的 bank-side 投影。Phase B 只学 per-position gate（=v1 全局 α 的
     升级版）时桥失败；加上 (I+P) 残差 projector 后桥成立。

2. **【二级证据】LPL 多轮动态 bank 在 2-hop 推理上确实工作** — Phase F
   anti-cheat (n=25 two-hop)：canonical Δ=−0.017 vs base，所有四种 ablation
   （shuffle layers / random bank / no cross-round read / K=1）把增益击穿
   14-40 倍。说明 bank 内容、跨轮读、layer 选址都 load-bearing。

3. **【负面发现】Phase A 的 simple_qa 增益是假象** — Phase F AC1（shuffle pause
   layer index）保留全部 simple_qa 增益 → 单跳事实回忆的 Δ 实际由"
   多算一遍 + pause 跳过本层"贡献，bank 内容无关。NegationQA 在所有配置
   下都伤害 base，说明 LPL 默认配置对否定语义有破坏。

4. **【方法论】Gate 0 bit-equal sanity 全程持有** — pause bias=−20 时
   24 个多样 prompt 上 max|Δlogits|=0.000e+00。

---

## 实验矩阵

| Phase | 设置 | 结果 | 状态 |
|---|---|---|---|
| 0 Plumbing | LPL patch 注入 Qwen3，bank+heads 实现，Gate 0 验证 | bit-equal 成立 | ✅ |
| A Frozen | force-pause @ {9,18,27} last-token, K=2, no train | simple_qa Δ=−0.30, two_hop Δ=−0.29, negation Δ=+0.03 → MOVEMENT_PASS 2/3 | ⚠️ 部分被 F 推翻 |
| F Anti-cheat | AC1-AC4 on Phase A | simple_qa 假，two_hop 真 | ✅ |
| F-25 | two_hop 扩到 N=25 | canonical 仅胜，AC 全垮 14-40× | ✅ |
| D Static bridge (naive) | 预填 Exp35b bank @ layer 9，无训练 | real=12.014, rand=11.994 ≈ 等价 | ❌ null |
| B Gate-only train | 训 bank_gate (92K)，80 步 | eval 几乎不动，pre/post Δ=−0.0001 | ❌ null |
| **B2 K-proj train** | **+ rank-64 (I+P) projector, 200 步** | **NLL 12.13→6.30 (Δ=−5.83), control rand=−0.02** | ✅ **核心** |

---

## 与 v1 公式的关系（用户原始问题的实证答案）

原计划写道："v1 的公式是没有办法直接用了"。Exp42 在工程上证明：
**v1 公式的结构（concat K,V into softmax + bank-side projection + gate）
是必要的；LPL 之于 v1 的真正升级只是把 "fact-write 一次性 (K,V) 预存"
换成 "h-store + 动态投影"，并把 "全局 α" 升级为 "per-position bank_gate"。**

> Phase B 的 null + Phase B2 的成功是 ablation 级别的证据：
> 缺 K-projector → 桥失败；补上 → 桥成立。

也就是说 v1 没死，是被"丢掉投影器"的简化版本带死的。复活只需要：

```python
bank_K^l = W_K^l · (h_b + P_K · h_b)     # P_K 低秩可学，零初始化
bank_V^l = W_V^l · (h_b + P_V · h_b)
```

实际 B2 只用了**共享的** P（同一个矩阵作用在 h_b 上，再让 W_K/W_V 各自分歧），
就拿到了 Δ=−5.83 NLL。后续可拆 P_K/P_V，跑分层 projector，扩到 N=10⁴ 预填。

---

## 文件结构

```
v1/experiments/atb_validation_v1/exp42_lpl/
├── attention_bank.py          # AttentionBank + LPLHeads
├── qwen3_lpl_patch.py         # decoder/attention monkey-patch
├── 01_phase_a_frozen.py       # Phase A
├── 02_phase_f_anticheat.py    # AC1-AC4 on Phase A
├── 03_two_hop_25.py           # Phase F 扩到 N=25
├── 04_phase_d_static_bridge.py# naive bridge null
├── 05_phase_b_train.py        # gate-only train null
├── 06_phase_b2_kproj.py       # 【成功】K-projector train
├── phase_a_results.json
├── phase_f_anticheat.json
├── phase_f_twohop25.json
├── phase_d_static_bridge.json
├── phase_b_train_results.json
├── phase_b2_kproj_results.json
└── EXP42_VERDICT.md           # 本文件
```

---

## 核心 numbers (Phase B2, 复现命令)

```
python3 v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py \
    --device mps --steps 200 --lr 2e-4 --rank 64
```

| Metric | Value |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 (frozen, bf16, MPS) |
| Bank preload | 512 Exp35b b-vectors @ layer 9, rescaled to L2=15 |
| Trainable params | 419,876 (P: 327,680 + bank_gate heads: 92,196) |
| Train set | 120 disjoint train-split (subj, rel, target) facts |
| Test set | 40 held-out test-split facts |
| Steps | 200 (33.7 s on M-series MPS) |
| Train loss curve | 14.06 → 6.45 |
| **base NLL on test** | **12.131** |
| **LPL+real bank+P, pre-train** | 12.127 (Δ=−0.004) |
| **LPL+real bank+P, post-train** | **6.305 (Δ=−5.826)** |
| LPL+random bank+P, post-train (control) | 12.112 (Δ=−0.019) |
| Bridge unlocked? | **TRUE** |

---

## 下一步建议（Phase C+）

1. **Phase B2-scale**：扩到 N=10⁴ 预填 / 1000 train items / per-layer P_K vs P_V
   分拆 / 多层 bank（不只 layer 9）。
2. **Phase B2-anticheat**：(a) shuffle bank rows after training → 应大幅退化；
   (b) train projector with random preload from start → 应学不到东西；
   (c) eval 时换全新（test b-vector）预填 → 测纯检索 vs 训练时共现的泛化。
3. **Phase C (halt+ACT)**：现在桥成立了，给 K_max≥3 + 学 halt head 更有动机。
4. **Phase E (interrupt demo)**：用 B2 训好的 projector 演示"程序中段
   inject 自定义 h_b 到 bank" 的 API。
5. **回退到原 v1 AttnNativeBank**：把 B2 的 P_K 集成回 `attn_native_bank.py`，
   补充 v1 套件原本缺的 learnable 投影。这是把 LPL 收获回灌 v1 主干的路径。

---

## 提交历史

- `ea9f1089` Phase A MOVEMENT_PASS
- `bb0c18c5` Phase F anti-cheat
- (this commit) Phase D + B null + **B2 K-projector success** + verdict

---
*Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>*
