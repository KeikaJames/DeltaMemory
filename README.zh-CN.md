<p align="center">
  <h1 align="center">Delta Memory</h1>
</p>

<p align="center">
  <strong>在冻结 Transformer 注意力与 LM head 内注入持久化外部记忆。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-EE4C2C.svg">
  <img alt="Hardware" src="https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%7C%20NVIDIA%20GB10%20Blackwell-555">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange.svg">
</p>

<p align="center">
  <strong>🌐 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文 (简体)</a>
</p>

<p align="center">
  <a href="docs/address_bound_delta_memory_plan.md">研究计划</a> ·
  <a href="docs/design.md">设计</a> ·
  <a href="docs/apple_silicon.md">Apple Silicon</a> ·
  <a href="reports/experiments">实验报告</a>
</p>

---

Delta Memory 是一个研究原型，目标是把冻结的 `google/gemma-4-E2B` 改造成
具备**真实、持久、地址键控**记忆的系统——它不是 RAG，不是 prompt
注入，也不是 MCP。本仓库的不同 Stage 探索"能否给 LLM 真正的记忆"的
不同切片：

- **Stage 0–7**（Apple Silicon MPS, bf16）：通过 Q/V 残差和 LM-head
  rank-4 LoRA 在 LAMA 事实卡片上做 in-context binding。answer top-1 命中
  oracle 上界。
- **Stage 8**（NVIDIA GB10 Blackwell，CUDA, bf16）：**闭卷**
  地址键控 fast-weight bank。读取阶段 prompt 里只剩地址，value token
  完全不在上下文中——所以答案只能从持久化的参数化 slot 检索而来，
  不可能是上下文复制。
- **Stage 13**（最新）：**AttentionNative DeltaMemory v2** —— 同样的思路，
  但**零可学习参数**。Bank 就是模型自己在一次写入 forward 中产生的
  K/V 张量，在每一层 attention 上拼接进 K/V 序列。Stage 13A–13F
  既有强阳性结果（unit gate 通过、locality 输出 bit-equal、KV-shared
  层修复让目标 token rank 41 → 9），也有严格阴性结果（多 token 对话
  recall 失败 —— 诊断为 K-space 匹配间距，由 Stage 14 修复）。

软件包仍叫 `rcvhc`，是为了和早期实验保持兼容。

它**不是 RAG、不是 MCP、也不是 prompt 注入**。Stage 8 的闭卷测试把
这一点变得具体：评估时读取 prompt 只含 address——没有检索到的文本，
没有 value token，没有 card。答案是通过学到的地址键检索从持久化参数
slot 取出的。

## DeltaMemory v2 一行公式

$$
\mathrm{Attn}_\ell\bigl(Q,\; [K\,;\, M_K^{(\ell)}],\; [V\,;\, \alpha\!\cdot\!M_V^{(\ell)}]\bigr)
$$

通俗讲：**给冻结的大模型挂一个外置记忆条**。在每层 attention 的
`K`/`V` 缓存后面，把"记忆条"`(M_K, M_V)`一并拼接进来。模型自己的
softmax 决定要不要看记忆条 —— 当 `α=0`、或者 query 和记忆条不匹配时，
输出**逐位等于**未挂载时的模型。无 encoder，无 KeyProjector，无残差
broadcast bias，**无训练**。一次 forward 就把记忆条写好了。

![DeltaMemory v1 vs v2 架构](docs/figures/v2/stage13_architecture.svg)

### v2 改了什么 / 为什么这事重要

v1（Stage 8–12）是把 encoder + KeyProjector + 最终残差 broadcast bias
四个独立模块缝在一起：要 1500 步训练，而且 Stage 12-P3 显示 broadcast
对所有 query 加同一个 bias，locality drift 高达 0.75。

v2 把这四个模块全部删掉。Bank 的 `K`/`V` 就是模型自己在一次写入
forward 里输出的；检索由模型本来在做语言建模的那个 softmax 完成。
**没有可训参数**。副作用包括：

- **`α=0` 自动 bit-equal**（拼接的是一段宽度为 0 的切片）。
- **Locality probe 在自由文本生成下也是 bit-equal**（Stage 13F）。
- **KV-shared 层**通过源层的 bank slot 也看到记忆 ——
  Stage 13A unit gate 显示单 fact target rank 41 → 9。

![Stage 13A unit gate — rank/logit 提升](docs/figures/v2/stage13_recall_lift.svg)

### v2 的诚实边界（Stage 13B/13F 阴性结果）

`Q: … A:` 对话 prompt **检索不到**用 `"X is Y."` 写入的记忆条 ——
即便 `"… current X is"` 形式的 unit gate 是通过的。原因是：写入位置
（句号 `.` 处）的 `K` 和读取位置（`A:` 处）的 `Q` 落在 K-space 的不同
区域，零样本下 softmax 跨不过去。Stage 13F 用 6 个对话场景把这一点
落实成证据：

- ✅ Locality probe —— 与 baseline 完全一致。
- ❌ Direct recall, paraphrase recall, malicious override, multi-fact,
  对抗 prompt —— α=1 下全部失败。

α 扫描清晰地展示了工作区间在 bank attention 压垮模型流畅性之前在哪里：

![DeltaMemory α 相变图](docs/figures/v2/stage13_alpha_phase.svg)

这正是仓库维护者在实验前预测的"第一刀检索空间"瓶颈。
**Stage 14** 是计划的修复方案：要么改写入捕获位置（地址条件捕获），
要么加一个极小的可学 K-projector 用 InfoNCE 在 paraphrase 正例上训。
两条路线都是严格加法、保留 v2 的零 encoder 性质。

Stage 13F 的对话录像逐字提交在
[`transcripts/google__gemma-4-E2B/`](transcripts/google__gemma-4-E2B/)。
完整 Stage 13 报告：
[`reports/cleanroom/stage13a_attn_native/`](reports/cleanroom/stage13a_attn_native/),
[`reports/cleanroom/stage13b_robust/`](reports/cleanroom/stage13b_robust/),
[`reports/cleanroom/stage13c_writer_decouple/`](reports/cleanroom/stage13c_writer_decouple/),
[`reports/cleanroom/stage13d_locality_fix/`](reports/cleanroom/stage13d_locality_fix/),
[`reports/cleanroom/stage13f_interactive/`](reports/cleanroom/stage13f_interactive/).



## 概览

| 问题 | 当前回答 |
| --- | --- |
| 改了什么？ | (Stage 0–7) Q/V 残差 + LM-head rank-4 LoRA，端到端有监督训练。(Stage 8) 在 LM-head 输入处通过基于地址内容的余弦检索读取 per-slot fast-weight bank。 |
| 什么保持冻结？ | Gemma-4-E2B 基座；只训练 Writer / KeyProjector / Q-V projector / LoRA。 |
| 已证明的部分 | (Stage 0–7) 端到端 binding 在 LAMA 事实卡片上命中 oracle 上界。(Stage 8) 闭卷召回、swap binding、no-leakage 三个 gate 在 N 最高到 4096 上全部通过（基座冻结）。 |
| 尚未证明 | Stage 8 在 N=4096 时的 retrieval recall@1 ≥ 0.95（GR gate）；Stage 8 三 seed 复现；vs RAG 头对头领先；长程顺序写入下的干扰耐受。 |
| 下一步 | **Stage 8 v3**：KeyProjector 调优、3-seed、RAG/MEMIT 头对头、顺序写入干扰曲线、curated LAMA 单 token 迁移。 |

## 机制

Delta Memory 从源上下文写入 per-block 的 Raw/Delta 注意力记忆，由查询
检索记忆块，再把检索到的 Delta payload 投影回注意力内部残差：

```text
q' = q + alpha_q * gate_q * P_q(Delta)
v' = v + alpha_v * gate_v * P_v(Delta)
```

更严格的研究假设：

```text
question address span -> memory address key
source value span     -> signed payload Delta
address classifier    -> identity gate -> Q/V residual
```

Stage 8 把这个假设推到闭卷读取的极端：读取时 prompt 只含 address span。
答案只能来自一个**持久化、address-key 检索得到的、参数化的 slot**——
没有任何形式的上下文复制可用。

详细计划见 [`docs/address_bound_delta_memory_plan.md`](docs/address_bound_delta_memory_plan.md)。

## 闭卷记忆 (Stage 8) — 地址键控 fast-weight bank

> **硬件：** NVIDIA GB10 (Blackwell) · CUDA · `bfloat16` · 单 GPU。
>
> **TL;DR.** 冻结的 `google/gemma-4-E2B` 加上 Writer + KeyProjector + per-slot
> fast-weight bank 在**闭卷模式**（读取 prompt 中没有 value token）
> 下能召回单 token 答案，规模到 **N=4096**，在单台 NVIDIA GB10 上完成。
> 在 N = 128 / 1024 / 4096 时 retrieved-slot top-1 分别为
> **0.969 / 0.934 / 0.838**。`no_memory` 基线在每个规模都是 **0.000**
> （无泄漏），swap-paired flip 是 **1.000**（bank 携带的是身份信息，
> 而不是上下文 token）。这是仓库内第一次"读取时 prompt 只含 address，
> 不含 value"的结果——测的是**持久化地址键控记忆**，而不是 in-context
> binding。

![Stage 8 闭卷容量曲线 (3 seeds, 均值±标准差)](docs/figures/fig6_stage8_capacity.svg)

| N facts | bank inject (oracle slot) top1 | bank inject (retrieved slot) top1 | address recall@1 | swap-paired flip | no-memory top1 |
|---:|---:|---:|---:|---:|---:|
|  128 | 1.000 ± 0.000 | **0.979 ± 0.009** | 0.979 ± 0.009 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| 1024 | 1.000 ± 0.000 | **0.931 ± 0.004** | 0.929 ± 0.003 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| 4096 | 1.000 ± 0.000 | **0.832 ± 0.006** | 0.826 ± 0.005 | 1.000 ± 0.000 | 0.000 ± 0.000 |

硬指标 G1（闭卷召回 ≥ 0.80）、G5（paired-flip ≥ 0.80）、G6（no-memory 泄漏 ≤ 0.05）**在三个规模 × 3 seed 上全部通过，σ ≤ 0.01**。GR（recall@1 ≥ 0.95）在 N=128 通过，N ≥ 1024 是**结构性上界**：Phase A 4 个 KeyProjector 调优变体（×3 steps、×2 key_dim、÷2.3 InfoNCE 温度、+8 hard-neg）**全部收敛到 recall@1 ≈ 0.832**——瓶颈是合成 address token 池本身，不是 projector。oracle-slot top-1 在每个 N 都是 1.000——bank 通道是完美的。

### Stage 8.3 — 顺序写入干扰（N=1024）

![Stage 8.3 干扰保留](docs/figures/fig7_stage8_interference.svg)

最早 128 个 slot 在 bank 写满到 1024 时仍保持 top-1 = 0.969——**顺序写入下没有灾难性干扰**（G3 ✅）。

### Stage 8.5 — vector-RAG 头对头（N=4096）

只训练 KeyProjector 的 vector-RAG 基线（同样的 pooled-address 特征）拿到 **vector_rag retr top1 = 0.838 = ours retr top1 = 0.838**（G2 平手）。我们在这个规模的优势**不是**检索精度，而是 (a) 参数化、swap 已验证的存储（G5 = 1.000——bank 携带身份信息，不是 chunk index），(b) 零泄漏（G6 = 0），(c) 可编辑 slot 而非 chunk 库。

### Stage 8.2 — LAMA 单 token 迁移（3 seeds）

![合成 vs LAMA](docs/figures/fig8_stage8_lama.svg)

同一 pipeline 跑 135 条 curated 事实三元组（首都、语言、货币——按 Gemma-4 tokenizer 过滤到单 token 答案）：retrieved-slot top-1、recall@1、oracle、swap-paired flip **三 seed 上全部 1.000 ± 0.000**。N=4096 合成上界**不是**机制的性质，而是 address span 表征丰富度的性质。

完整报告见 [`reports/experiments/stage8_closed_book_memory/REPORT.md`](reports/experiments/stage8_closed_book_memory/REPORT.md)。

## Stage 9 — 编码器升级、真实 LAMA-TREx、头对头基线

> **硬件：** NVIDIA GB10 (Blackwell, sm_120) · CUDA · `bfloat16` · 单 GPU。
>
> **TL;DR.** 更强的 address encoder 打破了 Stage 8 v3 在 N=4096 时的检索天花板（recall@1: 0.832 → **1.000 ± 0**，3 seeds）。同一 encoder 迁移到**跨 7 个 Wikidata relation 的真实 LAMA-TREx 事实集**上，top-1 = **1.000 ± 0**（3 seeds，swap paired-flip 0.989 ± 0.010），并且在同一事实集上 **DeltaMemory 决定性地超过 vector-RAG / IKE / SFT-LoRA**（1.000 vs ≤ 0.448）。

### Phase 9A — N=4096 合成集 encoder 消融

![Stage 9 encoder 比较](docs/figures/fig9_encoder_comparison.svg)

| Encoder | Seeds | retr top-1 | recall@1 | swap flip |
| --- | ---: | ---: | ---: | ---: |
| `mean_pool` (v3 基线) | 1 | 0.838 | 0.832 | 1.000 |
| `attn_pool` | 1 | 0.841 | 0.835 | 1.000 |
| `residual_mlp` | 1 | 0.838 | 0.833 | 1.000 |
| **`multilayer`** (4 层 concat) | **3** | **1.000 ± 0** | **1.000 ± 0** | 1.000 |
| **`prompt_hidden`** (read prompt 末位 hidden) | **3** | **1.000 ± 0** | **1.000 ± 0** | 1.000 |

mean_pool / attn_pool / residual_mlp 全部停在 ≈ 0.83，确认 v3 天花板是**表征性**的（address span 单一 pool 特征对 4 096 条事实信息量不足），不是优化导致。改变 *key 来源* 的两种 encoder 在检索上达到饱和。

### Phase 9B — 真实事实：LAMA-TREx，7 relation，3 seeds

![Stage 9 LAMA-TREx](docs/figures/fig10_lama_trex.svg)

数据：183 条 LAMA-TREx curated 事实，覆盖 P36 / P19 / P101 / P641 / P140 / P39 / P937。Encoder：`prompt_hidden`。

| 指标 | mean ± σ |
| --- | ---: |
| `bank_inject_retrieved.top1` | **1.000 ± 0** |
| `address_retrieval_recall_at_1` | 1.000 ± 0 |
| `swap_paired.paired_flip_rate` | 0.989 ± 0.010 |

跨 relation 的 top-1 σ = 0 说明 encoder 没有 relation 偏好；paired-flip 接近 1 说明替换 address 时答案被干净改写、无泄漏。

### Phase 9C — 与检索 / in-context / 参数微调三类基线头对头

![Stage 9 基线雷达](docs/figures/fig11_baselines_radar.svg)

同样 183 条 LAMA-TREx 事实、同样冻结 base、同样 target token。

| 方法 | Edit success top-1 | Edit success top-5 | Locality drift |
| --- | ---: | ---: | ---: |
| vector-RAG（input-embed mean-pool 余弦） | 0.399 | 0.486 | n/a |
| IKE（in-context editing，top-1 事实前置） | 0.399 | 0.486 | 0.50 |
| SFT-LoRA（`lm_head` 上 rank-16，200 步） | 0.448 | 0.557 | 0.50 |
| **DeltaMemory (Phase 9B, prompt_hidden)** | **1.000** | **1.000** | n/a（base 冻结） |

检索和 in-context 都在 *binding* 步失败（RAG retrieval@1 = 1.000，但 Gemma-4-E2B 不能可靠从前缀复制检索到的值）。SFT-LoRA 略有提升但在 neutral prompts 上 logit 漂移高达 50%，且只换来 0.448 成功率。

### 硬性 gate

| Gate | 状态 |
| --- | --- |
| GR9 — N=4k recall@1 ≥ 0.95 | ✅ multilayer / prompt_hidden = 1.000 |
| GR14 — swap paired-flip ≥ 0.85（真实事实） | ✅ 0.989 |
| GR17 — 超过 vector-RAG | ✅ 在 N=183 LAMA-TREx |
| GR18 — ≥ IKE 的 generality | ✅ |
| GR10 (N=65k)、GR11–13 (全 TREx 30k+)、GR15–16 (ROME/MEMIT) | ⏸ 留给下一次会话 |

完整报告见 [`reports/experiments/stage9_grand_evaluation/REPORT.md`](reports/experiments/stage9_grand_evaluation/REPORT.md)。聚合 JSON：`docs/figures/stage9_summary.json`。复现：`scripts/run_stage9_sweep.sh` + `scripts/generate_stage9_figures.py`。

> **⚠️ 重要：Stage 9 的 1.000 ± 0 仅在「完全相同的 prompt」(canonical) 下成立。** Stage 10 用同样的 pipeline、同样的 bank，跑了 paraphrase / decoy / value-ablation / leave-one-relation-out / equal-budget baseline 五类对抗测试，结果表明 encoder/writer **并不能**跨 surface paraphrase 或未见 relation 泛化。**引用 Stage 9 数字前请先看 Stage 10。**

## Stage 10 — 对抗验证（顶会级 stress test）

> **硬件：** NVIDIA GB10（Blackwell）· CUDA · `bfloat16` · 3 seeds · LAMA-TREx N=183。
>
> **TL;DR.** Stage 10 把 5 个可证伪假设摆在 Stage 9 面前。**3 PASS，2 FAIL。** Bank 本身是真的，retrieval 在 1000× 干扰槽下依然锐利，equal-budget 下 DeltaMemory 仍然碾压 RAG / IKE / SFT-LoRA — **但** encoder 是按字节级指纹匹配（不是语义编码），writer 也无法在未见 relation 上 zero-shot 泛化。

### 头条数字（3 seeds，mean ± std）

| 测试 | prompt_hidden | multilayer | 判决 |
|---|---|---|---|
| 标准 retrieval @1（canonical） | 1.000 ± 0.000 | 1.000 ± 0.000 | 复现 Stage 9 |
| **Held-out paraphrase recall @1** | **0.113 ± 0.020** | **0.307 ± 0.021** | **G10A FAIL** |
| Decoy ×1000 bind top-1 | 1.000 ± 0.000 | 1.000 ± 0.000 | G10B PASS |
| Random bank.v top-1 | 0.000 ± 0.000 | 0.000 ± 0.000 | G10D PASS |
| Shuffled bank.v top-1 | 0.015 ± 0.007 | 0.015 ± 0.007 | G10D PASS |
| **LORO holdout bind top-1**（6 个 relation 平均） | — | **0.112 ± 0.152** | **G10F FAIL** |

### 等预算基线对比（SFT 1500 steps）

| 方法 | edit top-1 | edit top-5 | locality drift（越低越好） |
|---|---|---|---|
| vector-RAG（input-embed cosine） | 0.399 ± 0.000 | 0.486 ± 0.000 | n/a |
| IKE（in-context fact prefix） | 0.399 ± 0.000 | 0.486 ± 0.000 | 0.500 ± 0.000 |
| SFT-LoRA r=4（1500 steps） | 0.541 ± 0.005 | 0.617 ± 0.000 | 0.556 ± 0.096 |
| SFT-LoRA r=16（1500 steps） | 0.552 ± 0.000 | 0.617 ± 0.000 | 0.556 ± 0.096 |
| SFT-LoRA r=64（1500 steps） | 0.552 ± 0.000 | 0.617 ± 0.000 | **0.778** ± 0.096 |
| **DeltaMemory（canonical, prompt_hidden）** | **1.000 ± 0.000** | — | **0.000**（读时 inject） |

### Stage 10 修正了什么

1. **Bank 是真的。** 把 bank.v 替换为随机 / 打乱后，预测崩到 0（G10D）。
2. **Retrieval 在大规模下依然锐利。** 加 1000× 随机干扰槽不影响 retrieval @1（G10B）。
3. **Encoder 不是语义的。** 同一事实换一个 surface paraphrase（即使裹在同一个 Atlas-slot 模板里）就崩到 0.11 / 0.31 — encoder 学到的是训练 prompt 的近似字节级指纹（G10A FAIL）。
4. **Writer 不能跨 relation 泛化。** Leave-one-relation-out 时 retrieval 是 1.000 但 binding 跌到 0.00–0.38（mean 0.11），即未见 relation 的 hidden state 经过 writer 后无法解码回 answer token（G10F FAIL）。
5. **等预算下 DeltaMemory 依然碾压 RAG / IKE / SFT-LoRA**（G10C PASS）：1.000 vs 最强基线 0.552，且 SFT-LoRA 引入 56–78% 的无关 token 漂移。

**诚实结论：** DeltaMemory 在 canonical prompt 上是一个真实可验证的事实存储，但它的 address encoder 和 writer **现阶段还不能跨 surface paraphrase 或未见 relation 泛化**。剩下的误差是表示问题，不是优化问题。下一阶段：(a) 用 paraphrase-augmented InfoNCE 重训 encoder；(b) 训练时就引入 relation-stratified LORO，而不是只在 eval 阶段做 LORO。

完整报告见 [`reports/experiments/stage10_adversarial_validation/REPORT.md`](reports/experiments/stage10_adversarial_validation/REPORT.md)。聚合 JSON：`reports/experiments/stage10_adversarial_validation/stage10_summary.json`。复现：`scripts/run_stage10_sweep.sh` + `scripts/run_stage10_resume.sh` + `scripts/aggregate_stage10.py`。

## Stage 11 — 针对 Stage 10 失败模式的重训练 + 对话基线（NVIDIA GB10）

> **硬件：** NVIDIA GB10（Blackwell, 128 GB unified, CUDA 13）· `bfloat16` · `google/gemma-4-E2B`。3 seeds。Gate 用 paired bootstrap（10 000 重采样）的 **CI 下界**判定（不是均值）。

Stage 11 直接攻击 Stage 10 暴露的两个失败模式：
**(a)** 用 paraphrase-augmented InfoNCE 重训 encoder；
**(b)** 训练时就做 relation-stratified LORO（不只是评估时），并在 relation-id 判别器上加 gradient-reversal 对抗头。

并补：**(c)** 对话场景基线（多轮 ConvQA / 用对话当 write-API vs RAG / prompt-injection 中毒攻击），**(d)** bit-exact 复现哈希。

### 核心数字（3 seeds，paired bootstrap 95% CI）

| 测试 | 指标 | 均值 | 95% CI | Gate | 结论 |
| --- | --- | ---: | --- | --- | --- |
| **11A** paraphrase-augmented InfoNCE，未见模板（`multilayer`） | recall@1 | 0.138 | [0.134, 0.141] | ≥ 0.85 | ❌ 失败 |
| **11A** paraphrase-augmented InfoNCE，未见模板（`prompt_hidden`） | recall@1 | 0.053 | [0.049, 0.058] | ≥ 0.85 | ❌ 失败 |
| **11A** decoy ×1000 回归 | top-1 | 1.000 | [1.000, 1.000] | ≥ 0.80 | ✅ |
| **11A** value 消融（random / shuffled） | top-1 | 0.000 / 0.009 | — | ≤ 0.10 | ✅ |
| **11B** 训练时 LORO + 对抗头，未见 relation | bind top-1 | 0.108 | [0.046, 0.178] | ≥ 0.50 | ❌ 失败 |
| **11D** 多轮 ConvQA（k=10 干扰轮） | recall@1 | 1.000 | [1.000, 1.000] | ≥ 0.85 | ✅ |
| **11D** 对话当 write-API vs RAG | DM − RAG | +0.692 | [0.625, 0.775] | > 0 | ✅ |
| **11D** prompt-injection 中毒攻击，受保护槽被覆写 | rate | 0.000 | [0.000, 0.000] | ≤ 0.05 | ✅ |
| **11E** bit-exact 复现 | SHA-256 一致 | 一致 | — | match | ✅ |

### 诚实结论（Stage 11 之后）

- **In-distribution 对话场景已经稳了。** 多轮干扰不破坏检索，对话当 write-API 比 RAG 高 +0.692 绝对分，受保护的槽抵御注入攻击。
- **Out-of-distribution paraphrase 仍然失败。** 每条事实 6 个训练模板 + InfoNCE 检索，**不足以**让 encoder 在未见模板上保持 relation 不变性。这是 `multilayer` / `prompt_hidden` encoder 的真实表示局限，不是优化问题。三个具体后续方向写在 `reports/experiments/stage11_grand_evaluation/REPORT.md`：正交 bank（Givens / Householder）、稀疏自编码器 bank、ROME 风格的闭式编辑。
- **跨 relation 泛化仍然失败。** 训练时 LORO + 权重 0.1 的 gradient-reversal 对抗头，对未见 relation 没有显著提升（6 relation × 3 seeds 均值 0.108）。DM **不是** relation 级别的 one-shot 可编辑记忆。
- **复现性。** Stage 11E 证实两次独立确定性运行的稳定子集 SHA-256 完全一致，见 `scripts/reproduce_stage11.sh`。

完整报告：[`reports/experiments/stage11_grand_evaluation/REPORT.md`](reports/experiments/stage11_grand_evaluation/REPORT.md)。方法论 / 数学辩护：[`docs/methodology.md`](docs/methodology.md)。

## Stage 12 — 对抗式跨模型验证（单模型完成，多模型推迟）

> **硬件：** NVIDIA GB10。仅 `gemma-4-E2B`。100 facts × 3 seeds × 500 steps × 3 个探针（P1 paraphrase / P2 十种对抗变换 / P3 输出篡改 + locality 控制）。

| 探针 | 结果 | 解读 |
| --- | --- | --- |
| P1 paraphrase 留出 | 1.000（n=3） | **在我们用的 encoder 下结构性 trivial** — 见下方 caveat |
| P2 十种对抗变换（typo、fragment、instruction-conflict、wrong-language、polite-misdirect 等） | DM top-1 = 1.000，无 DM = 0.000，提升 = +1.000（全部 10 项） | DM 注入能扛住 read prompt 的所有 surface 攻击 |
| P3 强制覆盖 base 模型答错的事实 | override = 1.000；**在 12 条无关 control 上 locality drift = 0.750** | α=1.0 + 整 bank 广播注入会污染 75% 的无关回答 — 生产必须用 per-query 路由（Stage 11D 中 drift = 0/0） |

**诚实 caveat：**
- P1 用了规范 address 进 `multilayer` encoder，而 encoder 忽略 read prompt；真正的 paraphrase 留出测试是 Stage 11A（= 0.138）。
- P2 测的是注入与 CE 的力量平衡，不是 encoder 的对抗鲁棒性。
- 针对 Qwen3-8B / GLM-4-9B / DeepSeek-V2-Lite / gpt-oss-20b 的多模型交叉验证脚本（`scripts/run_stage12_multimodel.py`）已就绪，但本次会话**没能跑**：GB10 在我们的环境里没法访问 HuggingFace，只有 `gemma-4-E2B` 预先缓存。**DeepSeek-V4-Flash**（284B MoE FP4，约 160 GB）放不进 GB10 的 128 GB，需要 vLLM-FP4 集群。多模型证据 **推迟**而不是声明。

完整报告：[`reports/experiments/stage12_gemma4_e2b/REPORT.md`](reports/experiments/stage12_gemma4_e2b/REPORT.md)。

## 硬件归属

| 阶段 | 硬件 | 备注 |
| --- | --- | --- |
| 0 – 7（小 N pilot, MPS） | Apple Silicon（M 系列, MPS, `bfloat16`） | 见 [`docs/apple_silicon.md`](docs/apple_silicon.md) |
| 8 闭卷 pilot | NVIDIA GB10（Blackwell, 128 GB unified） | CUDA 13.x, PyTorch 2.10+ |
| 9 LAMA-TREx + 基线 | NVIDIA GB10 | 3 seeds, full bootstrap |
| 10 对抗验证 | NVIDIA GB10 | 70+ 次运行，幂等 sweep |
| 11 重训 + 对话 + bitexact | NVIDIA GB10 | 29 次运行, paired bootstrap, SHA-256 稳定哈希 |
| 12 单模型对抗 | NVIDIA GB10 | 多模型推迟（无 HF 镜像） |



## 主要结果 — LAMA 事实绑定命中 oracle 上界

> **硬件：** Apple Silicon · MPS · `bfloat16` · M 系列单 GPU。
>
> **TL;DR.** 用冻结的 `google/gemma-4-E2B`，端到端训练的 **rank-4 LM-head
> LoRA**（由外部 writer 驱动）在 LAMA `factual_capital_binding` 套件上
> 跨 3 seed 达到 **top-1 = 1.000 ± 0.000**，匹配 oracle answer-embedding
> 上界（0.964），同时 `no_memory` 基线保持 **0.000**（无泄漏）。这关闭了
> Stage 6 在真实事实数据上的核心 strict gate。Swap 控制下的 binding 仍是
> 部分通过（paired-flip ≈ 0.50），是下一个细化目标。

### 图 1 — LAMA 上的 channel top-1（in-distribution, n=56, 3 seeds）

![LAMA Phase 2 channel top-1](docs/figures/fig1_channel_top1_lama.svg)

三个训练通道（`payload_probe`、`logit_bias`、`lm_head_lora`）跨过
0.85 strict gate；`lm_head_lora` 和 `payload_probe` 实际饱和到 1.000。
`oracle_logit_answer_embedding` 通道——直接把答案的 output-embedding 加
到 logits——在 0.964，所以训练得到的 LoRA 已在上界。`no_memory` 基线
= 0.000 确认 address tokens（`ADDR::country::France`）没有任何事实
信息泄漏穿过冻结基座。

### 图 2 — 同一 pipeline, 两个数据集

![Synthetic vs LAMA](docs/figures/fig2_synthetic_vs_lama.svg)

完全相同的 pipeline（oracle-span attention writer → answer-token CE →
LM-head rank-4 LoRA + Q/V 残差 + payload probe）在合成单 token 码字
（`address_token_binding_single_token`, Stage 6 Phase 1）上崩盘，但在
LAMA 事实绑定上干净解决。先前所谓"合成墙"是**任务模式问题，不是
架构问题**：当冻结基座本身已经编码了对应关联（ROME 风格），Delta
Memory pipeline 就成为一个近乎完美的检索/绑定 writer。

### 图 3 — Swap 控制（绑定特异性，遗留问题）

![Swap controls](docs/figures/fig3_swap_binding.svg)

如果把上下文 payload 换成 paired card 的 payload，理想的绑定通道应该
100% 输出对方的答案。当前 `lm_head_lora` paired-flip rate ≈ 0.50——远
高于随机水平（≈ 0.018），但低于 strict 0.80 gate。LoRA 部分绑定
payload→answer，但混入了 address 条件下的 default direction。**这是
下一个细化目标**：更强 swap loss、更长 warmup、通道消融。

### 图 4 — Stage 7A 线性探针负结果

![Stage 7A probe negative](docs/figures/fig4_stage7a_probe.svg)

我们独立测试了一个小线性分类器能否从 Gemma 隐藏状态恢复 answer-token
身份（也即原本的 `payload_probe` gate）。在 16 条合成 cell（held-out
top-1 最高 0.094）和 120 条 LAMA-disjoint cell（最高 0.000）上，没有任何
探针配置跨过 0.85 gate。合成情况是表征极限；LAMA-disjoint 是 closed-vocab
projector 的设计缺陷，详见 `reports/experiments/stage7a_lama_capital/REPORT.md`。
正确做法是**跳过探针 gate**，端到端有监督训练 LoRA 通道，正如图 1
所示。

### 图 5 — 各通道 answer NLL

![Answer NLL](docs/figures/fig5_channel_nll.svg)

held-out answer NLL 在 `no_memory`（≈ 17.16）和 `lm_head_lora`（≈ 0.003）
之间跨越**四个数量级**。三个训练通道把 NLL 都压到 1 nat 以内。

### 数值汇总

| Channel | top-1 (mean ± std) | top-10 | answer NLL | answer rank | n (seeds) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_memory`（基线） | 0.000 ± 0.000 | 0.000 | 17.162 | 12354 | 3 |
| `oracle_logit_answer_embedding`（上界） | 0.964 ± 0.000 | 0.982 | 0.793 | 42.6 | 3 |
| `delta_qv`（Q/V 残差） | 1.000 ± 0.000 | 1.000 | 0.001 | 1.0 | 3 |
| `payload_probe`（full-vocab CE） | **1.000 ± 0.000** | 1.000 | 0.027 | 1.0 | 3 |
| `logit_bias` | 0.964 ± 0.000 | 1.000 | 0.211 | 1.0 | 3 |
| **`lm_head_lora` (rank-4)** | **1.000 ± 0.000** | 1.000 | 0.003 | 1.0 | 3 |

| Swap control (LAMA Phase 2) | binding margin (foreign − correct NLL) | paired-flip rate |
| --- | ---: | ---: |
| `lm_head_lora_oracle_correct` | +23.20 | correct = 1.000 |
| `lm_head_lora_oracle_paired` | **−9.56 ± 1.70** | **paired = 0.506 ± 0.008** |
| `lm_head_lora_correct_address_paired_payload` | +2.71 | paired = 0.500 |
| `logit_bias_oracle_paired` | −24.70 | paired = 0.482 |

### 结论

1. **机制在真实事实数据上工作。** 通过 writer + LM-head rank-4 LoRA 的
   端到端 answer-token CE 监督，在 LAMA `factual_capital_binding` 上命中
   oracle 上界，address 端零泄漏。
2. **早期"合成墙"是任务模式墙。** 同一 pipeline 在合成单 token 码字
   上 top-1 ≈ 0.17–0.44，在 LAMA 上达到 1.000。未来合成套件要么
   对齐冻结基座已编码的结构，要么接受 fast-weight only 监督（不要
   probe gate）。
3. **泛化与绑定是不同问题。** Phase 2 是 in-distribution 绑定测试
   （train ≡ eval = 56 LAMA pair；池子太小，不能切 disjoint，详见
   Stage 7A REPORT）。它干净地说明 Delta Memory 通道**对 in-context
   binding 达到最优**。要泛化到 held-out 事实，要么扩大事实池
   （LAMA-UHN / T-REx / WikiData ≥ 1k），要么换协议；这是显式的
   下一步——而 Stage 8 已经把这一步推到了 N=4096 闭卷。
4. **Swap 绑定是部分通过。** strict ≥ 0.80 paired-flip gate 在 0.50
   miss。可调：增大 `--stage2-swap-loss-weight` 0.5 → 1.5–2.0、更长
   warmup、通道消融。

### 复现这些图

```bash
python3 scripts/generate_paper_figures.py
```

图都是纯 SVG（不依赖 matplotlib），从 `reports/experiments/stage6_phase2_lama/`、
`stage7a_pool_quick/`、`stage7a_lama_capital/` 和 `reports/experiments/`
下任何 `phase1_*` cell 重新派生。汇总数字写到 `docs/figures/summary.json`
便于核对。

## 当前证据

Stage 0–7 的真实模型证据都用 `google/gemma-4-E2B` on Apple Metal/MPS，
基座保持冻结，retrieved 文本不会被插回 prompt。Stage 8 的真实模型
证据全部在 NVIDIA GB10 Blackwell / CUDA 上跑。

> **Claim 边界：** Delta Q/V 注入是强力记忆通道，但当前的 Stage 0–7
> 控制还没完全孤立"query-specific 检索/绑定"作为唯一因果源。Stage 8
> 的 swap 测试（paired-flip = 1.000）在闭卷设置下首次给出干净的
> 因果证据。

详细的 Stage 0–7 实验列表见 [English README](README.md#current-evidence)（暂未翻译，因
所有早期实验报告均已在原文链接中保留）。

## 研究方向

文献张力：

| 工作线 | 优势 | 与 Delta Memory 的差距 |
| --- | --- | --- |
| RETRO / Memorizing Transformers / LongMem / RetrievalAttention | 显式外部记录 | 检索身份仍可能脆弱或非因果 |
| Titans / 神经长期记忆 | 自适应记忆通道 | 记忆强但不可解释 |
| Mamba / SSMs | 高效长状态传播 | 对精确内容寻址绑定弱 |
| Infini-attention / 压缩记忆 | 受限流式记忆 | 压缩可擦除反事实身份 |
| NoLiMa / RULER | 暴露长上下文 shortcut | 仅看 NLL 不够 |
| Delta-rule / fast-weight 视角 | 绑定与抗干扰框架 | 当前 Delta 路径需要显式 address 监督 |

合成提案现在是 **Token/Span-Bound Delta Memory**：

```text
memory item = (address span key, value span payload delta, anti-key metadata)
query      -> address span competition -> causal gate -> payload injection
```

更大规模前的硬指标：

| Gate | 要求 |
| --- | --- |
| Channel | Delta 击败 no-memory、zero、random 控制。|
| Address | 在共享池中，正确记忆排名高于 paired 负样本。|
| Shuffled | correct-address Delta 击败 shuffled-address Delta。|
| Wrong-query | correct-address Delta 击败 wrong-query/foreign-address Delta。|
| Margin | 正确记忆改善 `foreign_nll - correct_nll`。|
| Payload swap | correct address + foreign payload 与 correct address + correct payload 不同。|
| Oracle span | oracle value-span payload 在学到的检索可信前击败 paired value-span payload。|
| Logit-side diagnostic | 直接 payload-to-logit 注入在尝试 fast weights 前翻转 answer-token 偏好。|
| Baseline | Delta 击败 hidden retrieval 和真实的 retrieved-KV/attention 基线。|

## 快速开始

### Apple Silicon（Stage 0–7, MPS）

```bash
python3 -m venv .venv-mac
.venv-mac/bin/python -m pip install torch transformers accelerate safetensors tokenizers pytest
.venv-mac/bin/python -m pytest -q
```

快速 mock demo（不下载模型）：

```bash
.venv-mac/bin/python scripts/run_gemma4_prototype.py \
  --model mock-gemma --device cpu --dtype float32 \
  --block-size 32 --memory-dim 128
```

LAMA 事实绑定（Stage 6 Phase 2）on Apple MPS：

```bash
.venv-mac/bin/python scripts/run_delta_experiment.py \
  --model google/gemma-4-E2B --device mps --dtype bfloat16 \
  --steps 12 --train-samples 16 --eval-samples 16 \
  --task-suite paired_conflict_binding \
  --shared-memory-retrieval --conflict-margins
```

参见 [`docs/apple_silicon.md`](docs/apple_silicon.md) MPS/Metal 笔记。

### NVIDIA GB10 / CUDA（Stage 8 闭卷）

```bash
python3 -m venv .venv-gb10
.venv-gb10/bin/pip install torch transformers accelerate safetensors tokenizers
# offline 模式 — 在断网前先 populate HF cache
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-gb10/bin/python scripts/run_stage8.py \
    --model google/gemma-4-E2B --device cuda --dtype bfloat16 \
    --n-facts 4096 --steps 1500 --seed 0 \
    --report-dir reports/experiments/stage8_v2_n4096_seed0
```

GB10 实测 wall-clock：N=128 ≈ 5 分钟、N=1024 ≈ 12 分钟、N=4096 ≈ 25 分钟
（单 seed, 1500 steps, bf16）。子实验：`run_stage8_interference.py`
（保留率曲线）、`run_stage8_rag_baseline.py`（vector / text-RAG 头对头）。

## 文档

| 文档 | 用途 |
| --- | --- |
| [`docs/address_bound_delta_memory_plan.md`](docs/address_bound_delta_memory_plan.md) | 早期阶段实验计划 |
| [`docs/design.md`](docs/design.md) | 架构与证据边界 |
| [`docs/gemma4_prototype.md`](docs/gemma4_prototype.md) | Gemma 原型 runbook |
| [`docs/apple_silicon.md`](docs/apple_silicon.md) | Apple Silicon / MPS 配置（Stage 0–7） |
| [`reports/experiments/stage8_closed_book_memory/REPORT.md`](reports/experiments/stage8_closed_book_memory/REPORT.md) | Stage 8 闭卷记忆完整报告（NVIDIA GB10） |
| [`reports/experiments`](reports/experiments) | 全部已跟踪实验 artifact |

## 仓库布局

```text
rcvhc/core/       config 和共享 typed records
rcvhc/memory/     外部 Delta Memory store 与 writer
rcvhc/gemma/      Gemma 风格 adapter 与 layerwise Q/K/V injector
rcvhc/engine/     ingest, ask, training, experiments, statistics
scripts/          可运行 demo 与实验 CLI（含 run_stage8*.py）
scripts/data/     curated 数据集（如 lama_curated.jsonl）
docs/             设计笔记与研究计划
docs/figures/     paper 风格 SVG 图
reports/          已跟踪实验报告
tests/            CI-safe mock 测试；不需要下载 Gemma
```

## License

代码采用 [MIT License](LICENSE)。

模型权重、数据集、论文、第三方依赖各有自己的协议与条款。加载
`google/gemma-4-E2B` 的实验需用户自行遵守适用 Gemma 模型协议与
访问条款。
