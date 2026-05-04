# Phase V — 相关工作综述 (Related Work Survey)

> **目的**：在动手做 Phase W 实验之前，把同类研究全部看一遍，厘清我们方法（冻结权重 + 外置 KV bank + mHC column-cap + LOPI 正交投影 + V-scale）在文献图谱中的坐标，并列出每篇给我们的直接启发。
>
> **组织方式**：6 个桶（bucket），共 35 篇。每条目格式：(a) 它做了什么 (b) 与我们方法的关系 (c) 要不要试它的 trick / 如何用。

---

## 1. 外置记忆 / 检索增强（External Memory / Retrieval-Augmented, KV-side）

### 1.1 kNN-LM (Khandelwal et al., 2020) — arXiv:1911.00172

kNN-LM 在 transformer LM 的**输出端**做 kNN 插值：每次前向传播完成后，从外部数据库检索 k 个历史上下文的隐状态，用 softmax(−距离) 做一个额外的 token 分布，再与 LM 自身输出插值。全程不改变模型权重。我们的 KV-bank 思路上同属"加外置存储"，但 kNN-LM 挂在输出端、影响 token 分布，而我们挂在 attention 的 K/V 端、影响注意权重。kNN-LM 的插值系数 λ 需要验证集调参，等价于我们的 α。**Phase W.1 诊断可以参考其 λ 扫描曲线的 shape，看 drift vs α 是否也呈类似单调性。**

### 1.2 Memorizing Transformers (Wu et al., 2022) — arXiv:2203.08913

**与我们最近**。本文将长文档的历史 KV 对拼接进当前层的注意力，同时训练一个可学习的 gating 决定"用本地 KV 还是外置 KV"。在 PG-19 等长文档数据集上 perplexity 显著下降。关键差异：他们的 gate 经过训练；我们的 α 是冻结标量。他们的 ablation（第 4 节）给出了 gate-off 基线的 perplexity 代价，**这个数字可以当作 LOPI bank 不加 gate 的 upper-bound baseline**，与我们 W.1 "bank 拼接但 α 固定" 的性能做定量比较。必试：参照其 Figure 3 的 layer-wise gating 分布，检查我们的 attn_entropy_bank 是否有相同的 layer 峰。

### 1.3 RETRO (Borgeaud et al., 2022) — arXiv:2112.04426

RETRO 在 **decoder 每个 chunk** 通过 cross-attention 引入 kNN 检索文档块；模型参数只有 7B 但性能媲美 175B GPT-3。其核心是"检索增强的 cross-attn 层"（RETRO block），训练时联合优化检索和生成。我们不检索外部语料——我们的 bank 是人工植入的 fact；但 RETRO 关于"retrieval attention 的 head 数量、chunk size 对 perplexity 的影响"表格（Table 2）可以反映 bank slot 数 N 对信息利用率的 scaling 规律。**W.1 中 N=4/16/64/1024 grid 可对照 RETRO 的 chunk-size 实验。**

### 1.4 TRIME (Zhong et al., 2022) — [TODO: verify arXiv ID]

TRIME 提出**统一训练目标**，同时优化 token-level LM loss 和 retrieval loss（让模型表征在检索命中率上更好）。与 kNN-LM 和 Memorizing Transformers 的区别：后两者用预训练的 LM 表征做 kNN，不专门训练检索；TRIME 则对表征空间做联合调整。对我们的意义：提醒我们"冻结的 LM 表征不一定是最优的 KV bank key encoder"。**Phase W.2 中检查 LOPI 的 M_⊥/M_V 能量比时，可以借鉴 TRIME 的 key-quality 指标（检索命中率 @k）。不试（需要训练）。**

### 1.5 Unlimiformer (Bertsch et al., 2023) — arXiv:2305.01625

Unlimiformer 将 encoder-decoder 模型的 cross-attention offload 到 kNN 索引：每个 decoder attention head 的 query 在 CPU 或 GPU 的 kNN 索引里检索 top-k，把 kNN 距离当作 attention dot-product score 使用，支持 500k+ token 输入。实现上等同于"把原生 cross-attn 的 K/V 矩阵替换成动态检索的稀疏 K/V"。我们的 bank 拼接机制在概念上与此相同，但我们作用于 self-attention，且 bank 是小规模定制 fact，不是完整文档。**其 sub-linear kNN 查询实现（FAISS over CPU）可作为 Phase W.13 合成任务中大 bank (N=10000) 的工程参考。**

### 1.6 Landmark Attention (Mohtashami & Jaggi, 2023) — arXiv:2305.16300

本文在每个文本块末尾插入"landmark token"，训练 attention 先检索 landmark、再展开对应 block 的完整 KV，以随机访问方式处理任意长上下文。关键贡献：**retrieval 与 generation 共用同一 attention 机制**，无需独立检索模块。与我们的差异：landmark 通过有监督训练学得，我们的 bank key 是手工提供的 fact embedding。其 Figure 5（attention entropy vs block index）的分析方式可以直接移植到我们的 W.1.3 attn_entropy_bank 诊断。**看情况：如果 W.2 发现 LOPI ortho 有问题，可考虑 landmark-style 块边界 token 作为 bank pointer。**

### 1.7 GRIT-LM (Muennighoff et al., 2024) — arXiv:2402.09906

GRIT-LM 用 GRIT（Generative Representational Instruction Tuning）统一训练生成和嵌入能力：在 instruction 中区分两类任务，同一个模型既能生成文本又能出高质量嵌入向量（MTEB SOTA）。对我们的意义：我们需要把 fact key 编码成 bank query，理想情况下 key encoder 应与目标模型对齐。GRIT-LM 表明经过 instruction tuning 的 decoder 模型可以直接用作 encoder。**W.13 合成任务中如果需要对 sha-256→字符串 bank 做语义检索，可用 GritLM 7B 当 key encoder，无需额外训练。看情况。**

### 1.8 In-Context RALM (Ram et al., 2023) — arXiv:2302.00083

不修改 LM 架构，直接把检索文档拼到 prompt 前（prepend），即纯 in-context 的检索增强。对 GPT-2 到 GPT-J 等不同规模模型均显著改善 perplexity（WikiText-103 下降约 12-28%），且通用检索器（BM25、DPR）已足够有效。对我们的意义：这是我们方法的**最弱 baseline B0**——如果 RALM prepend 就能达到 implant rate = 90%，我们的 KV-bank 机制就没有价值。**必须在 W.6 和 W.10 中以 RALM-prepend 作为 B0 基线跑对比。**

### 1.9 Self-RAG (Asai et al., 2023) — arXiv:2310.11511

Self-RAG 训练 LM 动态决定是否检索（通过特殊的 [Retrieve] / [ISREL] / [ISSUP] / [ISUSE] reflection token），使模型能自适应地在"直接生成"和"检索后生成"之间切换，并对自己的输出做反事实评估。**其 special token 机制本质是学习了一个显式的 retrieval gate**——与我们冻结 α 的策略相对立。W.4 中若用 CAA 替换 LOPI，可对照 Self-RAG 的 reflection token，评估零训练 steering 能否模拟类似的"自适应检索"信号。**不试（需要训练）；但 reflection token 的评估指标（ISREL/ISSUP 准确率）可借鉴设计 W.2 的 Q2 指标。**

### 1.10 CRAG (Yan et al., 2024) — arXiv:2401.15884

CRAG 在 RAG 链路中引入轻量 **retrieval evaluator**，评估检索结果的质量并触发三种行为：直接使用、过滤后使用、Web 搜索补充。其核心洞见是"检索质量不稳定是 RAG 失败的主因"。对我们：我们的 bank 是人工构造、质量已知，理论上不需要 retrieval evaluator；但 CRAG 的"置信度阈值"设计可以指导 mHC 的 κ 参数选取——κ 过小相当于"总是全力信任 bank"，κ 过大相当于 CRAG 的 discard 动作。**W.1.4 DH3（N 小时 shield 是 no-op）与 CRAG 的"低置信度直接丢弃"在机制上对称，可互相印证。**

---

## 2. 知识编辑（Knowledge Editing, Weight-Side Baselines）

### 2.1 ROME (Meng et al., 2022) — arXiv:2202.05262

ROME 通过因果追踪（causal tracing）定位 GPT 中存储 factual association 的关键 MLP 层（通常为中间层），然后对该层的 W_fc 做秩-1 更新，使模型输出特定的"新事实"。在 CounterFact 数据集上 implant rate ≈ 99%，但顺序编辑后会出现严重 specificity 下降（副作用）。**ROME 是我们 W.6/W.14 的核心权重侧 baseline**；其 CounterFact 数据集（21919 条反事实关系）及评估指标（Efficacy Score、Paraphrase Score、Neighborhood Score）是我们必须复现的 benchmark 格式。必试（作为 W.14 数字基线）。

### 2.2 MEMIT (Meng et al., 2022) — arXiv:2210.07229

MEMIT 是 ROME 的批量版本：同时编辑多达 10000 条事实，通过在多个 MLP 层上分摊更新来减少干扰。解决了 ROME 顺序编辑导致"灾难性遗忘"的问题。其 MQuAKE 基准（多跳知识编辑）和 ZsRE 数据集（零样本关系提取）是我们 W.14 必用的评估集。**关键对比点：MEMIT 需要修改权重；我们在冻结权重下用 bank 植入，理论上不会有 specificity 代价。这一对比是我们工作的核心 claim，必须在 W.14 用相同数据验证。**

### 2.3 MEND (Mitchell et al., 2022) — arXiv:2110.11309

MEND 训练一个超网络（hypernetwork），输入"旧事实→新事实"的 gradient，输出参数增量，使编辑后模型的 output 改变。比 fine-tuning 快、比 ROME 更泛化，但**需要 meta-training**（在大量编辑样本上预训练超网络）。对我们：MEND 的超网络是"学习如何编辑"，我们的 bank+LOPI 是"冻结权重、用激活空间注入"。**不试（需要训练超网络）；但其 model-agnostic 设计思路（hypernetwork 跨模型迁移）可以启发 LOPI 的 cross-arch 泛化部分（W.1.1 5模型实验）。**

### 2.4 SERAC (Mitchell et al., 2022) — arXiv:2206.14795

SERAC 维护一个外部"编辑缓存"（scope-edit cache），配合一个轻量分类器决定每次查询是否命中编辑记录，如果命中则路由到专用的对立（counterfactual）小模型。不修改原始 LM 权重，但需要训练分类器和对立模型。从架构上看，SERAC 的"外部缓存+路由"与我们的"KV bank+mHC column-cap"有相似之处；关键区别是 SERAC 需要训练，我们冻结。**不试（需要训练路由器和对立模型）；其 scope 分类器的精度（precision/recall）是一个很好的"fact bank 命中率"指标模板，可以借鉴到 W.6 implant rate 的 breakdown 分析。**

### 2.5 GRACE (Hartvigsen et al., 2023) — arXiv:2211.11031

GRACE 将每次编辑存入一个外置的"episodic memory codebook"（类似 kNN 存储），推理时用余弦相似度决定是否从 codebook 取出覆盖原 LM 的预测。支持连续、终身编辑（lifelong editing），已编辑的条目不会干扰未编辑条目。**GRACE 与我们的 KV bank 最直接对应**：两者都是"冻结 LM，用外置存储覆盖特定知识"。主要区别：GRACE 在 logit 层操作，我们在 KV-attention 层操作。其 Appendix 中的 codebook 大小 vs 准确率曲线直接对应我们的 N（bank size）敏感性实验。**必须在 W.14 中以 GRACE 作为最强 KV-side baseline（B2）。**

### 2.6 R-ROME / EMMET (Gupta et al., 2024) — arXiv:2403.14236

本文统一了 ROME 和 MEMIT 的理论框架，称为"preservation-memorization objective"。ROME 使用等式约束（equality constraint），MEMIT 使用最小二乘约束；作者提出 EMMET（等式约束批量编辑），在 batch size 高达 10000 时与 MEMIT 性能相当。对我们：EMMET 的分析揭示了两种权重编辑的"有效编辑容量上限"（对应 MLP 矩阵的秩）。**这个上限与我们的 bank size N 上限分析平行——权重编辑的容量上限 ≈ MLP 秩，我们的上限 ≈ attn head 的 KV 维度。W.14 分析需要引用这个对比。**

### 2.7 AlphaEdit (Fang et al., 2024) — arXiv:2410.02355

AlphaEdit 在 ROME/MEMIT 的权重更新上额外施加约束：将扰动矩阵投影到"保留知识的零空间（null space）"，数学上证明投影后的更新不会改变已保留知识的输出。这与我们 LOPI 的 M_⊥ 正交投影**几乎是同一思路**，只是 AlphaEdit 应用于权重扰动，我们应用于 activation 投影。**这是 LOPI 理论正确性的一个有力先例证明（null space projection = specificity 保持）。W.2 的 Q1（M_⊥ 能量比 vs drift）可以明确引用 AlphaEdit 的证明思路。必看，强相关。**

---

## 3. Activation Steering / Representation Engineering（LOPI 的潜在替代品）

### 3.1 ITI — Inference-Time Intervention (Li et al., 2023) — arXiv:2306.03341

ITI 在推理时干预特定 attention head 的激活：用线性探针找到与"真实性"相关的 head，在那些 head 的激活上加一个沿"真实方向"的偏移量，方向来自对比 (honest vs dishonest) 样本对的激活差。在 TruthfulQA 上提升 LLaMA-7B 约 20%。对我们：ITI 选 head 的方式（按线性探针 F1 排序取 top-K heads）可以作为 LOPI 中 μ_arch（目标层选取）的**替代策略**——不用残差范数，而用 linear probe accuracy 选层。**W.2 Q3（μ_arch 选取的正确性）可以跑 ITI-style probe accuracy 作为对照指标。**

### 3.2 RepE — Representation Engineering (Zou et al., 2023) — arXiv:2310.01405

RepE 系统地研究"模型内部表征如何编码高级概念（诚实、情绪、道德等）"，提出用对比激活（positive vs negative 样本对）的 PCA 提取"representation direction"，再通过加/减 direction 在推理时控制模型行为。核心发现：在中间层的残差流上，很多行为方向是**线性可分的**。这为 activation steering 类方法提供了理论基础。对我们：LOPI 的正交投影实质上是在把 bank memory 投影到一个"先验方向的补空间"，与 RepE 找的 concept direction 性质类似。**W.4（CAA vs LOPI 对比）中可以用 RepE 的 PCA 可视化方法来展示 fact 向量在残差流中的方向，作为 LOPI 机理的可视化证据。**

### 3.3 ActAdd — Activation Addition (Turner et al., 2023) — arXiv:2308.10248

ActAdd 在特定层的 residual stream 里直接加上"激活差向量"（positive prompt − negative prompt 的激活之差），无需训练，即可在推理时改变模型行为（如让模型更话痨、更积极等）。测试于 GPT-2 到 GPT-J。方法极简：单次前向传播得到方向向量，推理时加到每个 token 位置。**ActAdd 是 W.4 的基础候选替代 LOPI 的方法之一**。其核心优势是零训练、跨 token 通用；劣势是方向向量是从 prompt pair 得到的，缺乏对"特定事实内容"的精确编码。**必试（W.4）：用 ActAdd 风格的方向向量替换 LOPI 投影，看 implant rate。**

### 3.4 CAA — Contrastive Activation Addition (Rimsky et al., 2024) — arXiv:2312.06681

CAA 改进 ActAdd：用**多对**对比样本（正例 - 负例）激活的均值差来计算更稳定的 steering vector，在 Llama 2 Chat 上做行为控制（减少权威主义、提升诚实性等）。其 ablation 表明 steering vector 在中间层（layer 15-20 for 7B）最有效，与 ITI 的发现一致。**CAA 是 W.4 的主要替代候选（核心实验）**。若 CAA 在 implant rate 上打过 LOPI，则在 W-Tune T1.4 后替换 LOPI。其 layer 敏感性结果（Figure 4）直接指导我们的 μ_arch 范围搜索。**必试（W.4）。**

### 3.5 Function Vectors (Todd et al., 2024) — arXiv:2310.15213

Function Vectors 发现 in-context learning 的任务信息（如"大写化"、"翻译"）被编码在少数 attention head 的输出中，表现为一个紧凑向量（FV）。FV 可以从 ICL context 中提取，然后在 zero-shot 设置下直接加到残差流里触发相同任务，无需 demonstrations。与 LOPI 的关系：LOPI 试图把"记住 fact X"编码到激活增量里，Function Vectors 已经证明"执行函数 f"可以压缩成单一向量。**W.2 Q1 的 M_⊥/M_V 分析可以引用 FV 的"任务向量" vs "内容向量"分离作为先例；W.4 CAA 实验中可用 FV 方式提取 fact-specific vector 作为 ablation。**

### 3.6 Linear Representation Hypothesis (Park et al., 2024) — arXiv:2311.03658

本文为"高级概念以方向向量形式线性表示"提供了严格的数学基础：用反事实语言（counterfactual language）给出"线性表征"的两种形式化，并证明它们与线性探针和模型 steering 分别对应。定义了一个"因果内积"（causal inner product），使几何度量（余弦相似度、投影）在语言结构上有语义意义。**这篇论文为 LOPI 的 M_⊥ 正交投影赋予了严格的理论解释**：在因果内积下，M_⊥ 是把 bank fact 投影到"不影响先验输出"的子空间。W.2 Q1 的理论动机可以直接引用 Park+ 2024。**必读，是 LOPI 的理论根基。**

### 3.7 Steering Vectors (Subramani et al., 2022) — arXiv:2205.05124

这是 activation steering 的早期工作之一：通过优化（梯度下降）找到一个向量，加到 LSTM/GPT-2 的隐状态后能以最大概率生成目标字符串。找到的向量在目标模型上可迁移，且有语义解释性（可用于分类和代码生成等）。历史意义大于直接应用价值。**对我们：确认"激活向量可以承载 factual content"这一假设早在 LSTM 时代就有实验支撑，降低 LOPI/CAA 类方法的理论风险。看情况，作为 related work 引用。**

---

## 4. Attention 谱控制 / 归一化（Attention Spectral / Normalization Control）

### 4.1 Sinkformer (Sander et al., 2022) — arXiv:2110.11773

Sinkformer 将 attention 矩阵的归一化从单次 softmax（行和=1）改为**Sinkhorn 迭代**（行和=1 且列和=1，即双随机矩阵）。通过多次交替行/列归一化使 attention 收敛到 doubly stochastic 状态。理论上防止 attention sink（某列权重过大）。**与我们的 mHC column-cap 是连续谱的两个端点**：Sinkformer 是全局迭代到稳态（完全对称），我们的 cap 只截断 bank 列的和到 κ（局部单侧约束）。可以把 mHC 看作"Sinkformer 的 budget-bounded 近似"。**W.1 诊断中 bank_col_sum 分布可以与 Sinkformer 的列和收敛曲线做类比。看情况：W-T1.4b 可以试一轮 Sinkformer-lite iteration 代替 hard cap。**

### 4.2 σReparam (Zhai et al., 2023) — arXiv:2303.06296

σReparam 对 Q/K 矩阵做谱归一化（spectral normalization），用 σ(W)−1 重参数化，使 attention logits 的量纲稳定，避免 attention collapse（softmax 后概率集中在一个 token）。在 ViT 训练中显著提升稳定性和最终性能。与我们的 QK-Norm 关注点相同，但方法不同（σReparam 直接约束矩阵谱范数，QK-Norm 归一化向量 L2 范数）。**对 W.1 的意义：若 mHC 对 bank K 矩阵的列和只是 symptom 而非 cause，真正的问题可能是 K/Q 谱范数失控——可以在 W.1 diagnostics 中加一列 spectral_norm(bank_K)。**

### 4.3 Scaling Vision Transformers to 22B (Dehghani et al., 2023) — arXiv:2302.05442

本文将 σReparam 应用于超大规模 ViT（22B 参数）的工程实践，报告了 attention logit 不稳定是 22B 规模训练 NaN 的主因，σReparam 是关键的稳定化手段。同时提出了 head-parallel 训练等技巧。对我们的意义：**在大规模模型上，K/Q 谱不稳定是真实工程问题，不只是理论担忧**。这支持了 mHC column-cap 的工程必要性，为 W.1 的立论提供了宏观背景。**不试（工程规模不同）；但其 attention logit 统计（mean/std vs training step）的监控方式可以移植到 W.0.2 diagnostics。**

### 4.4 QK-Norm (Henry et al., 2020) — arXiv:2010.04245

QK-Norm 对 Q 和 K 向量分别做 L2 归一化后再点积，控制 attention logit 的幅值，防止 softmax 饱和。这是一个极简的操作，已被 Llama 3、Gemma 等模型采用（有的用 RMSNorm over Q/K）。**我们的 Gemma-4-E2B 已经使用 v_norm（等价于 V 向量 RMSNorm），这与 QK-Norm 的 V 侧版本直接对应**。W.1.4 DH2（无 v_norm 模型 drift 更高）的理论背景正是 QK-Norm 的延伸。**必须在 W.1 REPORT 中引用此文解释为何 v_norm 是关键变量。**

### 4.5 DeepNorm (Wang et al., 2022) — arXiv:2203.00555

DeepNorm 提出在 Post-LN transformer 中对残差连接加一个常数 α 的缩放（x → α·x + sublayer(x)），同时用 β 初始化 sublayer 参数，理论上使梯度方差在深层有界。**对我们的关系**：LOPI 在某些设置下等价于给 residual stream 加一个外置增量（Δx = LOPI(bank)），DeepNorm 已经证明"残差增量有界"是训练稳定的充分条件。**这支持 LOPI 的"冻结 + 小幅 activation 注入"思路的稳定性。W.2 中 residual_norm 诊断（W.0.2 第 5 类信号）可以引用 DeepNorm 的方差上界作为期望值基准。**

### 4.6 HyperConnections (Sun et al., 2024, ByteDance) — arXiv:2409.19606

HyperConnections 将残差连接替换为"超连接"：每一层可以从**所有历史层**动态加权聚合，兼具 Dense connection 和 Residual connection 的优点，同时避免 seesaw effect（梯度消失 vs 表征坍塌的权衡）。在 LLM 预训练和 ViT 上均显著优于残差基线。**与 LOPI 的关联**：LOPI 的 μ_arch 层选取相当于在"哪一层注入 memory 最有效"上做单点决策；HyperConnections 提供了多层加权的视角。**W.2 Q3（μ_arch 是否一致）可以参照 HyperConnections 的跨层贡献权重，分析 LOPI 注入应否分散到多层。看情况（W-Tune 阶段可试多层注入变体）。**

### 4.7 mHC — Multi-Head Concentration (DeepSeek, 2025/2026) — [TODO: verify arXiv ID]

mHC 是我们自研方法（在 DeepSeek 架构基础上提出），对 bank 列的 attention 权重和施加上限 κ，防止 bank slot 占主导导致原生 attention 被稀释。该方法直接是 Phase W.1 的被试方法，此处列出是为了完整性和与上述文献的定位对比。**与 Sinkformer 的关系**：Sinkformer 约束整个注意力矩阵为双随机，mHC 只约束 bank 子矩阵的列和上界，是局部、单侧的轻量约束，计算开销 O(N)。**与 σReparam 的关系**：σReparam 约束 Q/K 谱范数（乘法），mHC 约束列和（加法截断）。两者可以叠加。

---

## 5. MoE Attention 与 Per-Expert 控制（MoE & Per-Expert Control）

### 5.1 Mixtral 8×7B (Jiang et al., 2024) — arXiv:2401.04088

Mixtral 采用稀疏 MoE FFN：每个 token 由 router 选 top-2（共 8 个）专家，理论参数 46.7B 但激活参数仅 12.9B。Router 使用 softmax 后 top-k 选通（hard gating）。**对 W.5 per-expert cap**：Mixtral 的 router 没有 per-expert capacity constraint（即没有 token-dropping），与 Switch Transformer 不同。**验证我们的 per-expert cap 公式时，需要区分"有 capacity limit 的路由"（Switch/Expert Choice）和"无 limit 的 top-k 路由"（Mixtral）两种情况，分别对应不同的 col-sum 期望值推导。**

### 5.2 Switch Transformer (Fedus et al., 2021) — arXiv:2101.03961

Switch Transformer 是第一个成功的 sparse MoE transformer LLM：每个 token 只路由到 top-1 专家，配合 **expert capacity** 约束（每个专家最多接受 capacity_factor × (tokens/num_experts) 个 token，超出则 token overflow 直接走残差）。**Capacity 约束是 W.5 per-expert cap 公式的直接先例**：cap 值 = capacity_factor × E[bank_slots_per_head]。**必须在 W.5 REPORT 中对照 Switch 的 capacity factor 分析来验证我们的 κ 选取。**

### 5.3 Expert Choice (Zhou et al., 2022) — arXiv:2202.09368

Expert Choice 反转 token-to-expert 的选择方向：让每个专家主动选取自己最感兴趣的 top-k tokens（而不是 token 选专家）。这天然地保证 expert load balance（每个专家恰好处理 k 个 token），消除 Switch 的 token overflow 问题。对我们：Expert Choice 的"专家选 token"逻辑等价于"bank slot 主动选哪些 query 向量进行响应"，与 mHC 的 column-cap 逻辑方向相反但互补。**W.5 中如果发现某些 bank slot 几乎不被关注（dead slot 问题），可借鉴 Expert Choice 的 load-balance 思想给 bank 加一个 min-selection 约束。**

### 5.4 Soft MoE (Puigcerver et al., 2024) — arXiv:2308.00951

Soft MoE 消除了 MoE 的离散 routing，改用**软分配**：每个 token 先通过可学习权重被"软混合"成若干 slot 表示，再分别由专家处理，最后逆变换回 token 空间。消除了 token dropping 和 load imbalance 问题，同时在 ViT 和 LLM 上优于 hard MoE。**与 LOPI 的类比**：LOPI 的 γ_t（derivative gate）在 [0,1] 之间连续变化，等价于在"使用 bank 记忆"和"忽略 bank 记忆"之间的软选择——与 Soft MoE 的 soft dispatch 机制同构。**W.2 Q2（γ_t 分布是否退化到 0/1）可以引用 Soft MoE 的 dispatch weight 分布作为对照。**

### 5.5 DeepSeek-V3 / MLA (DeepSeek Team, 2024) — arXiv:2412.19437

DeepSeek-V3 提出 Multi-head Latent Attention（MLA）：将 K/V 压缩为低秩潜表示（down-projection 到 dim=512，再 up-project 回 head_dim×num_heads），显著减少 KV cache，同时配合 FP8 混合精度和 Multi-Token Prediction 辅助目标。**对 W.5**：MLA 的 down-up projection 结构意味着 bank 的 K/V 必须先做同样的 down-projection 才能对齐 model 的 KV 空间——这是 per-expert cap 公式中 bank_col_sum 计算的一个边界情况，需要在 W.5 中特别标注 MLA 架构的处理方式。**W.1 模型阵列中若加入 DeepSeek 模型，此点是关键工程细节。**

### 5.6 Qwen-MoE / Qwen2-MoE (Qwen Team, 2024) — arXiv:2408.07178 [TODO: verify arXiv ID]

Qwen-MoE 采用细粒度专家（fine-grained expert）：把传统 MoE 的 N 个大专家拆分为 mN 个小专家，每个 token 激活 top-k 个小专家，保持相同激活参数但提升专家多样性。同时引入共享专家（shared expert），始终激活，减少冗余学习。**对 W.5**：shared expert 机制类似于我们的"native attention 始终激活，bank 为可选增量"的设计——这为 W.5 的 per-expert cap 公式中"native vs bank 分离"提供了架构层面的先例支撑。

### 5.7 OLMoE (Muennighoff et al., 2024) — arXiv:2409.02060

OLMoE 是完全开源的稀疏 MoE LLM（1B active params, 7B total），包括训练代码、数据、权重和详细的 router 行为分析。其 router 统计（每个专家被选中的频率分布、expert specialization 指标）是最透明的开源参考。**W.5 per-expert cap 公式正确性验证可以直接用 OLMoE 的 router checkpoint 复现 col-sum 期望值**，因为 OLMoE 开放了完整 checkpoints 和路由日志。**必试（W.5 验证工具）。**

---

## 6. 跨架构校准 / 参数高效方法（Cross-Arch Calibration / Parameter-Efficient）

### 6.1 IA³ (Liu et al., 2022) — arXiv:2205.05638

IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）为每一层的 K、V 和 FFN 中间激活引入可学习标量向量，乘（element-wise）到这些激活上，只有 ~0.01% 参数量，却在 few-shot fine-tuning 上媲美全参微调。**与我们 V-scale 的关系极为密切**：V-scale 对 bank V 矩阵乘一个标量（等价于 IA³ 的 l_v），但我们的 l_v 是冻结的、按 L2 范数计算，而 IA³ 是训练的。**W.2 Q1 的 V-scale ablation 应当明确引用 IA³ 作为"训练版 V-scale"的 baseline，论文中的 ablation（Table 5）给出了去掉 l_v 的 性能代价，为我们提供了预期的 sensitivity 量级。必读。**

### 6.2 Prefix Tuning (Li & Liang, 2021) — arXiv:2101.00190

Prefix Tuning 在每一层的 KV cache 前缀拼接可训练向量（prefix tokens），只训练 prefix 参数（~0.1% 参数），冻结其余模型。GPT-2 和 BART 上接近全量微调性能。**与我们的 bank 机制完全同构**：我们的 bank KV 就是 inference-time 的 prefix，只是我们的 prefix 不是训练得来，而是从 fact 的表征直接提取（冻结）。**Prefix Tuning 已经证明"前置 KV 可以有效影响生成行为"，这为我们方法的有效性提供了最直接的 feasibility 证据。必须在论文中作为主要比较对象之一（W.10 对照实验 B1）。**

### 6.3 Prompt Tuning (Lester et al., 2021) — arXiv:2104.08691

Prompt Tuning 仅在输入 embedding 层添加可训练的 soft token（不做层级拼接），训练开销更低，随模型规模增大性能接近全量微调（11B T5 上 ≈ 全量 fine-tune）。与 Prefix Tuning 的区别：只在 input 层，不在每层 KV 做 prefix。**对我们**：Soft Prompt 是我们方法在"输入层注入"维度的对照——如果只加一层 prefix token 就够用，我们的多层 bank 就是 overkill。**W.10 的 B1 baseline 可以包含一个"Soft Prompt prepend"变体，与 bank KV 注入做对比，检验"层级注入 vs 输入层注入"的效果差。**

### 6.4 BitFit (Zaken et al., 2021) — arXiv:2106.10199

BitFit 只微调 transformer 的**偏置项（bias terms）**，参数量约 0.08%，在 BERT-based 模型的 GLUE benchmark 上接近全参微调。发现 bias 参数承载了大量"任务适配"信息，而 weight 矩阵更像通用特征提取器。**对我们**：如果 LOPI 的 gauss noise 项（A2/A3 variant）可以理解为对 bank embedding 的随机 bias 调整，BitFit 的发现支持"小幅 bias 扰动就能改变任务特性"的假设。**W.2 Q1 可以引用 BitFit 来解释为什么 gauss 分量能单独影响 drift（即使与 ortho 无关联时）。不试（不在我们冻结权重路线上）。**

### 6.5 Soft Prompts as Memory (Lester et al., 2021 / Power et al., 2022) — 见 Prompt Tuning，扩展视角

从"记忆"的视角重新解读 Soft Prompt：prompt 向量可以被视为压缩进 embedding 空间的"外置记忆"，模型通过 attention 与这些向量交互来"读取"记忆。这与我们的 KV bank 在设计意图上完全相同，区别仅在于：(a) Soft Prompt 只在输入层，bank 在每个注意力层；(b) Soft Prompt 是训练的，bank 是冻结的；(c) Soft Prompt 的"记忆内容"是隐式压缩的，bank 的 KV 是显式的 fact embedding。**这个视角为 Phase W.13（合成字典任务）的设计提供了理论框架：测量 bank 的"explicit memory capacity" vs soft prompt 的"implicit memory capacity"。**

---

## 末段：启发列表

### ✅ 必试（必须在 Phase W 中跑 / 对照）

| Trick / 方法 | 来自 | Phase W 子相位 | 动作 |
|---|---|---|---|
| CAA steering vector 替换 LOPI | Rimsky+ 2024 | W.4 | 用对比激活均值差向量代替 LOPI 投影，比较 implant rate |
| ROME / MEMIT 数字作为权重侧 baseline | Meng+ 2022 | W.14 | 在 CounterFact / ZsRE / MQuAKE 上跑 ROME/MEMIT，作为 baseline 比较 |
| GRACE 作为外置存储 baseline (B2) | Hartvigsen+ 2023 | W.14 | 用 GRACE codebook 做 KV-side 知识编辑对照 |
| In-Context RALM prepend (B0) | Ram+ 2023 | W.6, W.10 | 把 fact 直接 prepend 到 prompt 的简单 baseline |
| Prefix Tuning style bank (B1) | Li+ 2021 | W.10 | 训练版 prefix KV 作为"有训练的上限" |
| IA³ ablation as V-scale reference | Liu+ 2022 | W.2 Q1 | 引用 IA³ Table 5 的 l_v 去除代价估计 V-scale 贡献量级 |
| OLMoE router log 验证 W.5 公式 | Muennighoff+ 2024 | W.5 | 用开源 router checkpoint 复现 col-sum 期望值 |
| ActAdd 风格向量（单对样本版） | Turner+ 2023 | W.4 | 在 CAA 的 ablation 中包含单对 vs 多对方向向量比较 |

### ⚙️ 看情况（W-Tune 各环可能引入）

| Trick / 方法 | 来自 | 触发条件 | 动作 |
|---|---|---|---|
| Sinkformer-lite column 迭代 | Sander+ 2022 | 若 W.1 mHC hard-cap 效果不稳定 | 在 W-T1.4b 试 2 次 Sinkhorn 迭代代替截断 |
| ITI-style head 选取替换 μ_arch | Li+ 2023 | 若 W.2 Q3 显示残差范数 peak ≠ μ_arch | 用线性探针 F1 重新选注入层 |
| GRIT-LM 作为 bank key encoder | Muennighoff+ 2024 | 若 W.13 合成任务 key collision 率高 | 用 GritLM 编码 fact key 替换原生隐状态 |
| HyperConnections 多层注入 | Sun+ 2024 | 若 W.2 发现单层注入 μ_arch 方差大 | 在 W-T 阶段试"每层按权重分散注入" |
| RepE PCA 可视化 | Zou+ 2023 | W.4 分析阶段 | 用 PCA 可视化 fact 方向在残差流中的线性可分性 |
| FV（Function Vector）提取 fact 向量 | Todd+ 2024 | W.4 ablation | 比较 FV-style ICL 提取 vs LOPI 正交投影的效率 |
| DeepNorm 方差上界参照 | Wang+ 2022 | W.0.2 diagnostics 设计 | 在 residual_norm 监控中加入 DeepNorm 预期上界标注 |

### ❌ 不试（不在冻结权重路线 / 需要训练 / 超出算力）

| 方法 | 原因 |
|---|---|
| MEND | 需要训练超网络（meta-learning on edit corpus），与冻结权重思路矛盾 |
| SERAC | 需要训练分类器和对立模型，属于有参系统 |
| Self-RAG | 需要训练 reflection token，是 fine-tuning 范式 |
| TRIME | 需要联合训练检索和生成目标（representation learning） |
| Full Sinkformer | 全局双随机迭代会改变 native attention 的分布，破坏先验 |
| BitFit（作为参数更新方法） | 修改权重，我们不训练 |
| Prompt Tuning（训练版） | 需要训练 soft tokens，只有推理时固定 bank 的变体才符合我们的设定 |
| Scaling ViT 22B 技术 | 规模不匹配（22B vs 我们 0.5B-9B 测试集），工程目标不同 |

---

*生成日期：2025-06。总条目 35 篇（含子节 6.5 计为独立视角条目）。如需更新，请在每节末标注 [TODO: verify arXiv ID] 的条目用实际 arXiv 编号替换。*
