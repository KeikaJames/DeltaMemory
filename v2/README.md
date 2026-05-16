# v2 — Hippocampus-style Native LLM Memory (HNM)

> **状态**：v2 是 active 主线；`v1/` 是 archive（保留以复现历史）。

> ⚠️ **2026-05 更新（pivot）**：e11 噪声鲁棒性实验（wave-3 @ L9 + wave-5 @ L21）
> 已**证伪**「bank content carries the information that is read out」这一
> 原始论点。随机 Gaussian / 单行复制 / 常量向量 bank 给出与真实 bank
> *相同甚至更大* 的 Δ NLL。当前 working interpretation：v2 实质是一个
> **rank-64 K-projector 残差适配器，以 AttentionBank API 作为表面形式**，
> 而非 hippocampus 式内容检索系统。详细见
> [`verdicts/V2_FINAL_VERDICT.md`](verdicts/V2_FINAL_VERDICT.md) §1b
> 与 [`verdicts/E01_VERDICT.md`](verdicts/E01_VERDICT.md)。
> 余下决定性实验：e10 top-K（content-vs-capacity 判别）、e13 multi-task
> （capacity transfer 检验）。

## 起源

v1 探索了 ATB / mHC / RCV-mC / Manifold / Sinkhorn / Hyper-Connections 等多条
路线，最终在 Exp42 LPL Phase B2 拿到关键实证：

> 在 Qwen3-4B-Instruct-2507（整模型冻结）上，把 Exp35b 的 512 个 MEMIT
> 残差 b-vector 预填入 layer 9 AttentionBank，再训一个 rank-64 (I+P) 残差
> K/V 投影器（420K 可训练参数，200 步），test NLL **12.13 → 6.30**
> （Δ=−5.83）；同体量 random bank + 同一训好的投影器仅 Δ=−0.02。

详见 `v1/experiments/atb_validation_v1/exp42_lpl/EXP42_VERDICT.md`。

v2 把这个发现放进一个完整的、可被人和模型共同使用的**原生记忆机制**框架：

## 三大核心机制

1. **K-Projector 桥**（已 PoC，需大规模反作弊）
   - bank-side 可学 (I+P) 把异质 latent 对齐到目标 layer 的 QK 语义空间
   - 没有它，v1 公式跑不通；有了它，静态长期记忆库可被 frozen 模型直接读

2. **双通道注入**：
   - **AI 自动 pause-retrieve**：pause head 学习何时跳过当前层 attention，
     把 hidden 写入 bank；下一轮同层 attention 把它当 KV 读
   - **人工 interrupt**：`v2.core.interrupt_api.interrupt(bank, ...)` 公开
     API，外部程序可在任意 (round, layer, position) 注入 latent

3. **长短期共存**：
   - LT (long-term)：preload Exp35b 的 b-vectors，frozen
   - ST (short-term)：pause-write 的工作记忆，FIFO 上限 C
   - 同一 AttentionBank 容器，下一轮 attention concat 一并读

## 与 v1 的对照

| | v1 AttnNativeBank | v2 HNM |
|---|---|---|
| 写 | 一次 fact-prompt forward 预存 (K, V) | h-store + 运行时投影；pause-write 跨轮累积 |
| 读 | concat (K, V) 进 softmax，全局 α | 同结构，bank-side learnable (I+P)，per-position bank_gate |
| Round | 单 round | K_max 轮 + ACT halt |
| 注入 | 不支持 | 双通道（auto-pause + interrupt API） |
| 长短期 | 仅长期（fact-write） | 长短共存 |

## 仓库结构

```
v2/
├── core/                       # 单源真理：所有 experiments 共享
│   ├── attention_bank.py       # 长短期混合 bank
│   ├── qwen3_lpl_patch.py      # 双通道 hook
│   ├── runtime.py              # K_max 多轮 + halt
│   ├── kproj.py                # (I+P) 投影器
│   ├── retrieval.py            # topK 检索（cosine / dot / learned）
│   ├── interrupt_api.py        # 公开人工注入接口
│   ├── load_model.py
│   ├── data_io.py              # bank.pt 读、relation split
│   └── eval_lib.py             # NLL / PPL / acc helpers
├── experiments/
│   ├── e01_anticheat_b2/                 # B2 falsifiers H1-H10
│   ├── e02_scale_matrix/                 # N_preload×N_train×layers×lr×steps
│   ├── e03_capability_drift/             # WikiText-103 + lm-eval-harness
│   ├── e04_act_halt_kmax/                # K∈{2,4,8} + halt
│   ├── e05_cross_model/                  # Qwen3 / Llama / Mistral
│   ├── e06_relation_disjoint_ood/
│   ├── e07_per_layer_kproj/
│   ├── e08_interrupt_api_demo/
│   ├── e09_v1_resurrect_attn_native_bank/
│   ├── e10_topk_retrieval/
│   ├── e11_dual_channel/
│   ├── e12_long_short_coexistence/
│   ├── e13_multi_task_capability/
│   ├── e14_pause_head_train/
│   ├── e15_ponder_curriculum/
│   ├── e16_bank_capacity_forgetting/
│   ├── e17_negation_robustness/
│   ├── e18_chained_2hop/
│   └── e19_seed_replication/
├── methodology/
│   ├── V2_METHODOLOGY_DEBATE.md
│   └── V2_DIFFERENTIATION.md
├── tech_debt/
│   └── V1_CLOSEOUT.md
└── verdicts/
    ├── E01_VERDICT.md ... E19_VERDICT.md
    └── V2_FINAL_VERDICT.md
```

## 复现 B2

```bash
python3 v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py \
    --device mps --steps 200 --lr 2e-4 --rank 64
```

## 主要中央问题

> A frozen LLM, augmented with (i) per-position learnable pause head,
> (ii) per-layer K/V projector over a shared AttentionBank, and (iii)
> multi-round halt mechanism, can integrate hippocampus-style long-term
> memory and within-inference working memory through native attention—
> no fine-tuning of base weights, no prompt rewriting, no external retriever.

10 个廉价解释（H1-H10）必须分别证伪。详见 `methodology/V2_METHODOLOGY_DEBATE.md`。
