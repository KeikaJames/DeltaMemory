# ECOR：能量守恒正交旋转注入算子

**文档版本**: v1.0 · Phase X.7  
**作者**: KeikaJames, 2026-05-04  
**状态**: 实验性 (opt-in); 默认关闭，等待 W-T3.6 消融验证  
**代码参考**: `deltamemory/memory/lopi_inject.py`

---

## 第一节：用户提出的 ECOR 原始公式

### 背景

Mneme 的 LOPI（动态正交投影注入）算子在注意力层的值空间做加法注入：

$$V_{\text{out}} = V_{\text{ctx}} + \gamma_t \cdot w_\ell \cdot M_\perp$$

其中 $M_\perp = M_V - \operatorname{proj}_{V_{\text{ctx}}}(M_V)$ 是记忆库值向量在上下文值向量正交补空间中的分量，$\gamma_t$ 是导数门控，$w_\ell$ 是层高斯权重。

**用户提案**：将加法替换为旋转，使得 $\|V_{\text{out}}\| = \|V_{\text{ctx}}\|$（能量守恒）：

$$\theta = \frac{\pi}{2} \tanh(k \cdot \gamma_t \cdot w_\ell \cdot \alpha)$$

$$V_{\text{out}} = \cos\theta \cdot V_{\text{ctx}} + \sin\theta \cdot \frac{M_\perp}{\|M_\perp\|} \cdot \|V_{\text{ctx}}\|$$

### 代码草图（用户原始版本）

```python
theta = (math.pi / 2) * torch.tanh(k * gamma_t * w_ell * alpha)
M_unit = M_perp / M_perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
M_scaled = M_unit * V_ctx.norm(dim=-1, keepdim=True)
cos_t = torch.cos(theta).unsqueeze(-1)
sin_t = torch.sin(theta).unsqueeze(-1)
V_out = cos_t * V_ctx + sin_t * M_scaled
```

**直觉**：当 $\theta = 0$（门控关闭）时 $V_{\text{out}} = V_{\text{ctx}}$；当 $\theta \to \pi/2$ 时完全旋转至 $M_\perp$ 方向，且 $\|V_{\text{out}}\| = \|V_{\text{ctx}}\|$（Givens 旋转保范性）。

---

## 第二节：九条合理异议

### 异议 1：$\theta \to \pi/2$ 时 $V_{\text{ctx}}$ 被彻底抹除

当 $s = \gamma_t w_\ell \alpha$ 较大（$k=3$ 时 $s > 1$ 即饱和），$\tanh(ks) \to 1$，则 $\theta \to \pi/2$，$\cos\theta \to 0$。此时上下文信息被完全清空，注入的内容成为唯一输出。在 $\alpha \in [2, 8]$ 的范围内，这正好是 R-3 中观测到注入过度（catastrophic drift）的区间，ECOR 在该区间的行为比加法式 LOPI **更激进**，而非更温和。

### 异议 2：范数守恒可能不是注意力值空间的正确不变量

注意力层的输出 $O = \operatorname{softmax}(QK^\top/\sqrt{d}) \cdot V$ 对 $V$ 的线性变换敏感，但并不要求 $\|V\|$ 守恒。Transformer 残差流的内在尺度由 LayerNorm 负责调节。强制守恒 $\|V_{\text{out}}\| = \|V_{\text{ctx}}\|$ 在理论上是一个额外约束，未经验证是否对下游语义有益，甚至可能与后续 FFN 的输入分布假设产生冲突。

### 异议 3：$\|M_\perp\| \to 0$ 时方向噪声爆炸

当记忆库贡献微弱（低 $\alpha$，或 $M_V$ 与 $V_{\text{ctx}}$ 高度平行），$M_\perp \approx 0$，单位化 $M_\perp / \|M_\perp\|$ 将数值噪声放大到单位球面。一旦沿噪声方向旋转，注入完全与记忆内容无关，行为不可预测。原始公式依赖 `clamp_min(1e-8)` 抑制除零，但任何极小 $\|M_\perp\|$ 都会使方向单位向量随机化。

### 异议 4：$\tanh(ks)$ 在 $k=3$ 时过早饱和

$s$ 的典型范围：$\alpha \in [0.25, 4]$，$\gamma_t \in [0, 1]$，$w_\ell \in (0, 1]$，故 $s \in [0, 4]$。$\tanh(3 \times 4) = \tanh(12) \approx 1$。实际上 $s > 0.5$ 时（$k=3$）$\theta$ 已经超过 $\pi/2 \times 0.9$。这意味着 $\theta$ 在绝大多数有实际意义的运行区间内都趋近于最大值，$k$ 参数实际上失去了调节作用，ECOR 退化为固定旋转 $\pi/2$。

### 异议 5：未修复 R-3 的根因（低 $\alpha$ 下 $M_\perp$ 方向误差）

R-3 消融的核心发现：**正交投影**（$M_\perp = M_V - \operatorname{proj}_{V_{\text{ctx}}}(M_V)$）在 $\alpha \in [0.25, 2]$ 范围内**增加**中性文本漂移，因为此时 $M_\perp$ 的方向本身就带有噪声（投影算子对低信噪比输入不稳定）。ECOR 的旋转仅替换了注入的"量"，并未修复注入的"方向"——沿着噪声方向旋转与沿着噪声方向加法同样有害，甚至因能量守恒约束而更难恢复。

### 异议 6：per-head 与 per-vector 粒度不明确

注意力值张量通常为 $(B, H, T, D_{\text{head}})$。"旋转在 $D_{\text{head}}$ 子空间中进行"（per-head）与"在整个 $D = H \times D_{\text{head}}$ 空间中进行"（per-vector）是两种不同的几何操作，对应不同的范数守恒语义和实现方式。原始提案未指定，而这两种选择在梯度传播和数值稳定性上有实质差异。若 per-head，则每头的 $\theta$ 可能不同，模型行为难以解释；若 per-vector，需将 $(B, H, T, D_{\text{head}})$ reshape 为 $(B, T, D)$，影响 KV-cache 兼容性。

### 异议 7：梯度饱和抑制 $\gamma_t$ 信号

即使在冻结 LLM（只训练 Mneme 参数）的情形下，$\gamma_t$ 仍可能通过端到端链条接收梯度（例如在训练 $\alpha$ 或门控参数时）。$\tanh'(x) = 1 - \tanh^2(x)$，当 $|x| > 2$ 时梯度几乎为零。加上 $k$ 放大，梯度饱和在 $s \approx 0.5$ 时已发生，而加法式 LOPI 的梯度是 $s$ 的线性函数，无饱和问题。这使得 ECOR 在任何需要优化 $\gamma_t$ 或 $w_\ell$ 的场景（如 W-T3.6 自适应门控实验）中学习信号稀疏。

### 异议 8：「正交旋转升级」表述夸大了数学改进

将 ECOR 称为对加法的"升级"在数学上不诚实：旋转和加法改变的是注入操作所属的**代数群**——加法在向量空间中是阿贝尔群操作，Givens 旋转是 $SO(D)$ 群操作。二者没有自然的"强弱"之分，只有不同的几何假设。在范数守恒的场合旋转更自然；在线性叠加的场合加法更自然。注意力值空间没有先验证据支持哪种约束更优，称旋转为"升级"是营销语言而非技术声明。

### 异议 9：与 ROME 式 rank-1 值投影更新的潜在干扰

ROME（Meng+ 2022）通过直接修改 FFN 权重矩阵写入事实记忆，其更新在值空间形成 rank-1 外积 $\Delta W = v u^\top$。若 Mneme 同时使用 ROME writer（`rome_writer.py`）和 ECOR 注入，旋转操作可能与 ROME 的 rank-1 更新方向产生耦合：旋转将 $V_{\text{ctx}}$ 旋离 ROME 写入的方向，削弱 ROME 的事实召回效果。加法式 LOPI 的向量叠加与 ROME 更新正交（不同机制），而旋转则在同一方向空间中竞争。此干扰在 `tests/test_rome_writer.py` 中未被覆盖，需专项测试。

---

## 第三节：改进版本——五重保障

基于上述九条异议，改进版本引入五重保障：

### 保障 1：最大角度上限（max_theta_frac）

将 $\theta$ 上限设为 `max_theta_frac · π`（默认 $1/3 \Rightarrow 60°$）而非 $\pi/2$：

$$\theta = \left(\frac{\pi}{3}\right) \tanh(k \cdot s)$$

此时 $\cos(\pi/3) = 0.5$，$V_{\text{ctx}}$ 的贡献永远不会低于 $50\%$，消除异议 1。

### 保障 2：软混合滑块（soft_blend）

```python
V_blend = (1 - soft_blend) * V_add + soft_blend * V_rot
```

`soft_blend=0`（默认）退化为纯加法，`soft_blend=1` 为纯 ECOR，中间值为插值。W-T3.6 消融在 $\{0, 0.16, 0.33, 0.5, 0.66, 0.83, 1.0\}$ 上扫描，找到最优点。这同时解决异议 2（不强制守恒）和异议 8（不声称"升级"，只是可选项）。

### 保障 3：方向稳定性回退（direction_eps）

```python
direction_stable = (norm_m / norm_v) > direction_eps  # 默认 1e-3
return torch.where(direction_stable, V_blend, V_add)
```

当 $\|M_\perp\| / \|V_{\text{ctx}}\| < \epsilon$ 时，自动回退到加法路径，消除异议 3（方向噪声）和异议 5（低 $\alpha$ 下 $M_\perp$ 退化）。

### 保障 4：per-head 模式（per_head=True，默认）

通过在 `dim=-1`（头维度）上做范数归约，旋转在每个注意力头的 $D_{\text{head}}$ 子空间内独立进行，与注意力机制的多头语义一致，消除异议 6 的粒度歧义（默认选择 per-head，可通过 `per_head=False` 退出）。

### 保障 5：自适应 k（adaptive k）

`k=None` 时使用保守默认值 `k=1.0`（异议 4 中的早饱和问题在 $k=1$ 时 $s=2$ 才达到 $\tanh \approx 0.96$，大幅延迟饱和区间）。profiler 或超参搜索可通过 `ECORConfig(k=...)` 覆盖，消除异议 7 中固定大 $k$ 导致的梯度饱和。

---

## 第四节：与文献的关联

### Givens 旋转

ECOR 是 Givens 旋转（Givens, 1954）在值向量子空间中的应用。Givens 旋转是 QR 分解的基本操作，在数值线性代数中已有严格的误差分析。ECOR 可视为在 $\text{span}(V_{\text{ctx}}, M_\perp)$ 二维平面内的旋转，保持平面之外的分量不变。

### Falorsi et al. (2018)

Falorsi 等人研究了在神经网络中参数化旋转群（$SO(n)$）的方法，指出旋转参数化相比加法具有更好的归纳偏置（当任务本身具有旋转等变性时）。注意力值空间是否具有此类等变性尚未证明，但 ECOR 的动机与 Falorsi+ 的框架一致。

### Cohen & Welling (2016) — 等变 CNN

Cohen & Welling 的等变卷积网络将旋转等变性内建为结构约束。ECOR 的 per-head 旋转可类比为在特征空间中引入轻量等变约束，与该工作的精神相通，但 ECOR 针对的是**注入**而非网络权重本身。

### RoPE（Su et al., 2021）

旋转位置编码（Rotary Position Embedding）也在 $Q$ 和 $K$ 向量上施加 Givens 旋转来编码位置信息，其数学形式与 ECOR 高度相似：

$$Q' = R(\theta_{\text{pos}}) \cdot Q, \quad K' = R(\theta_{\text{pos}}) \cdot K$$

ECOR 可被视为在**值向量**上施加内容驱动旋转（$\theta$ 由记忆门控决定），与 RoPE 在查询/键上施加位置驱动旋转形成对偶关系。两者都利用旋转保范性，可能具有协同效应（未验证）。

---

## 第五节：升格为默认的验收准则

### W-T3.6 预登记实验设计

**扫描网格**：

| 变量 | 取值 |
|------|------|
| `soft_blend` | $\{0, 0.16, 0.33, 0.50, 0.66, 0.83, 1.0\}$ |
| 模型 | GPT-2, LLaMA-3-8B, Gemma-2-9B（共 3 个） |
| $\alpha$ | $\{0.25, 0.5, 1.0, 2.0, 4.0\}$（共 5 个） |
| 随机种子 | $\{0, 1, 2\}$（共 3 个） |

**总计**：$7 \times 3 \times 5 \times 3 = 315$ 个运行。

**评估指标**：与 `soft_blend=0`（加法基线）相比，以下指标的配对差：
- 主指标：Counterfact 事实召回准确率（CR↑）
- 副指标：中性文本困惑度漂移（PPL-drift↓）

**统计检验**：
- 对每个 (模型 × $\alpha$) 组合，用 3 个种子做配对 Wilcoxon 符号秩检验
- **ECOR 胜出条件**（同时满足）：
  1. 在 $\geq 3/5$ 的 $\alpha$ 水平上，$\exists$ `soft_blend` $\geq 0.5$ 使主指标改善 $p < 0.05$
  2. 在 $\geq 2/3$ 的模型上满足条件 1
  3. 在所有有显著差异的运行中，PPL-drift 不显著恶化（单边检验 $p > 0.05$）
- 若不满足，ECOR 保持 opt-in 标志，不升格为默认

**预登记文件**：`reports/prereg/ecor_v10_prereg.md`（待创建于 W-T3.6 开始前）

---

## 第六节：代码参考表

| 组件 | 文件 | 行号（近似） |
|------|------|------------|
| `ECORConfig` dataclass | `deltamemory/memory/lopi_inject.py` | 45–63 |
| `_align_gamma` broadcast helper | `deltamemory/memory/lopi_inject.py` | 67–94 |
| `lopi_inject` 主函数入口 | `deltamemory/memory/lopi_inject.py` | 98–185 |
| 加法路径早返回（位等同保证） | `deltamemory/memory/lopi_inject.py` | 143–146 |
| ECOR 旋转路径 | `deltamemory/memory/lopi_inject.py` | 148–185 |
| 方向稳定性门控 | `deltamemory/memory/lopi_inject.py` | 156 |
| 软混合 | `deltamemory/memory/lopi_inject.py` | 180 |
| 方向回退 `torch.where` | `deltamemory/memory/lopi_inject.py` | 183 |
| `__init__.py` 导出 | `deltamemory/memory/__init__.py` | 5–6 |
| 单元测试（8 个用例） | `tests/test_lopi_inject.py` | 全文 |
| 烟雾测试脚本 | `experiments/X7_ecor_smoke/smoke.py` | 全文 |

---

## 第七节：W-T3.6 TODO 列表

- [ ] **预登记**：在实验开始前提交 `reports/prereg/ecor_v10_prereg.md`，锁定超参和检验准则
- [ ] **接线**：在 `attn_native_bank.py` 中调整 `apply_lopi` 调用点，传入 `v_ctx_readout` 并使用 `lopi_inject` 替代裸加法，验证 `tests/test_lopi_module.py` 全部通过
- [ ] **消融扫描**：实现 `experiments/W_T36_ecor_ablation/run.py`，执行 315 个运行并生成 `reports/W_T36_ecor_ablation/results.csv`
- [ ] **ROME 干扰测试**：添加 `tests/test_ecor_rome_interference.py`，验证同时使用 ECOR 和 ROME writer 时，ROME 事实召回不显著下降（异议 9）
- [ ] **梯度饱和分析**：在 `experiments/W_T36_ecor_grad/` 中记录不同 $k$ 值下 $\gamma_t$ 梯度范数，确认 $k=1$ 的保守性
- [ ] **自适应 k 接线**：探索 `lopi_profiler.py` 是否可输出推荐的 $k$ 值（基于 $\|M_\perp\| / \|V_{\text{ctx}}\|$ 分布），填充 `ECORConfig.k`
- [ ] **文档更新**：实验结论写入本文档第五节，更新验收结果
- [ ] **升格决策**：若满足第五节验收准则，在 `LOPIConfig` 中添加 `ecor: ECORConfig = field(default_factory=ECORConfig)` 并将 `apply_lopi` 接线升为正式 default
