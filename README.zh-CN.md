<p align="center">
  <h1 align="center">Mneme</h1>
</p>

<p align="center">
  <strong>把外置 K/V 记忆直接注入冻结 Transformer 的 attention。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-EE4C2C.svg">
  <img alt="Hardware" src="https://img.shields.io/badge/Apple%20MPS%20%7C%20GB10%20CUDA-bf16-555.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange.svg">
</p>

<p align="center">
  <strong>语言：</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="docs/design.md">设计</a> ·
  <a href="docs/apple_silicon.md">Apple Silicon</a> ·
  <a href="docs/HISTORY.md">阶段历史</a> ·
  <a href="reports/cleanroom">实验报告</a>
</p>

---

Mneme 是一个研究原型，目标是在**冻结 LLM** 上加一层持久化外置记忆。
每层 attention 被拼进一份外置 K/V bank，读阶段 prompt 里只有问题，基座
权重始终冻结。Phase R+ 的标准栈把 bank 与一个免训练的注入包装层（Dynamic
LOPI v3.4）和一次性残差 profiler（U-LOPI Phase S）配成一套，使得同一份代
码在 Gemma / Qwen3 / GLM-4 / Llama / GPT-2 上无需手调 α。它**不是 RAG**，
**不是 prompt insertion**，也**不是权重编辑**。

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamemory import (
    AttnNativePatcher, fresh_bank, write_fact,
    profile_residuals, LOPIConfig,
    save_bank, load_bank,
)

model_name = "google/gemma-4-E2B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

# 1) bank + 每层 attention patcher（包在冻结 LLM 之外）。
patcher = AttnNativePatcher(model)
bank = fresh_bank(model)
bank.lopi_cfg = LOPIConfig(enabled=True, profile_mode="auto")  # Phase S 自动校准

# 2) U-LOPI 冷启动：一次性残差 profile -> 按架构的 Z-score 基线。
prof = profile_residuals(model, tok)            # forward-only，权重 bit-equal
bank.attach_lopi_profile(model, tok)            # 把 profile 绑到 bank.lopi_state

# 3) 写入一条事实。
write_fact(patcher, bank, tok,
           write_prompt="Fact: Python was created by Ada Lovelace.",
           fact_id="py_ada", address="Python")

# 4) 挂上 bank 解码。
read_prompt = "Q: Who created the Python programming language?\nA:"
with patcher.patched(), patcher.injecting(bank, alpha=1.0), torch.no_grad():
    out = model.generate(**tok(read_prompt, return_tensors="pt"), max_new_tokens=8)
print(tok.decode(out[0], skip_special_tokens=True))

# 5) 持久化（schema "ulopi_v36"）；load_bank 会同时还原 profile 与 V-scale 配置。
save_bank(bank, root="./banks", model_name=model_name)
```

`bank.attach_lopi_profile(...)` 是 `profile_residuals` 的一层薄包装：把
profile 绑到 `bank.lopi_state` 并校验层数与 bank shape 一致。
`LOPIConfig(enabled=False)` 时 merged-softmax 分支与 v3.1 老路径逐位等价；
`α=0` / 空 bank 与原模型逐位等价。

## 架构

### AttnNativeBank

冻结 LLM 的外置 attention bank。每个非共享 attention 层 ℓ，bank 在模型的
`num_key_value_heads` 分辨率下存 pre-RoPE K 和 post-norm V，并直接拼进
attention：

$$
\mathrm{Attn}_\ell\bigl(Q,\; [K\,;\, M_K^{(\ell)}],\; [V\,;\, \alpha M_V^{(\ell)}]\bigr)
$$

bank 没有任何可训练参数；检索空间**就是**模型自己的 K-space，attention
softmax 充当对比引擎。GQA / MQA 复用模型自带的 `repeat_kv`；KV-shared 层
（如 Gemma 4）在读阶段查询其源层的 bank slot，所以每个 attention 层都看
得到 bank。

* 文件：[`deltamemory/memory/attn_native_bank.py`](deltamemory/memory/attn_native_bank.py)
* Patcher：`AttnNativePatcher`。辅助：`fresh_bank`、`write_fact`、`forward_with_bank`。
* Bit-equal sanity：`tests/test_attn_native_bank.py`。

### Dynamic LOPI v3.4

merged-softmax 分支的免训练包装层，把

`out_bank = weights[..., T:] @ (alpha * mv_e)`

替换为三个独立、可配置开关的分量：

$$
\mathrm{out\_bank}_{\mathrm{LOPI}} \;=\; \gamma_t \cdot w(\ell, t) \cdot M_\perp,
\quad M_\perp = M_V - \mathrm{proj}_{V_{\mathrm{ctx}}}(M_V)
$$

* **Orthogonal Novelty**（`M_perp`）丢掉与上下文 V 平行的冗余分量。
  （v3.4 默认关闭；v3.3 ablation 时可打开。）
* **Adaptive Layer Gaussian** `w(ℓ, t)` —— 关于层号的 Gaussian，中心 `μ_t`
  由上一步残差范数驱动，宽度 `σ_t` 受运行中 mHC 最大-σ 稳定信号收缩。
* **Derivative Gate** `γ_t = sigmoid(k · (‖Q_t − Q_{t-1}‖₂ − θ))`：话题
  稳定时静音，话题切换时打开。

`LOPIConfig(enabled=True, orthogonal=False, gaussian=True, derivative=True)`
是 v3.4 默认；`enabled=False` 与 `α=0` 都与原模型逐位等价。

* 文件：[`deltamemory/memory/lopi.py`](deltamemory/memory/lopi.py)
* 公开符号：`LOPIConfig`、`LOPIState`、`apply_lopi`、`derivative_gate`、
  `layer_gaussian_weight`、`orthogonal_novelty`。

### U-LOPI Phase S

v3.4 将 `norm_base = 10.0` 设为固定常数，按 Gemma-4-E2B 校准；其它模型族
的 residual 尺度相差 10–100×，因此在跨架构上表现受限。Phase S 以一次冷启
动 profile 取代该全局常数：在小型中性语料上前向，统计 `‖hidden_states[ℓ]‖₂`，与 bank 一并持
久化。深度信号在 Z-score 空间计算，`μ_t` 自动锚定到该架构的尖峰层：

$$
z_\ell(t) \;=\; \frac{N_t(\ell) - \mu_{\mathrm{base}}(\ell)}{\sigma_{\mathrm{base}}(\ell) + \varepsilon},
\qquad
\mu_{\mathrm{arch}} \;=\; \arg\max_\ell\, \sigma_{\mathrm{base}}(\ell)
$$

只是一次 `output_hidden_states=True` 的前向，不引入任何 `nn.Parameter`，
LLM 权重前后逐位一致（由 `test_lopi_profiler.py::test_profile_does_not_mutate_weights`
验证）。`LOPIConfig(profile_mode="auto")`（默认）下，`norm_base` /
`mu_low` / `mu_span` 在运行时被忽略；`profile_mode="static"` 用于回归 v3.4
完全一致的行为。

* 文件：[`deltamemory/memory/lopi_profiler.py`](deltamemory/memory/lopi_profiler.py)
* 公开符号：`LOPIProfile`、`profile_residuals`、`default_profile_corpus`、
  `save_profile`、`load_profile`。
* 跨架构覆盖：`tests/test_lopi_universal.py`（Gemma / Qwen3 / GLM-4 /
  Llama / GPT-2 的 shape 与 bit-equality 检查）。

### 持久化（Phase R-6）

按版本、内容寻址的 bank 存储：`<root>/<model_safe>/<config_sha>/`，其中
`config_sha` 是 bank 相关配置（架构 shape + LOPI cfg + bank 温度 + shield
开关 + V-scale 校准）的 sha256。每层 `M_K`/`M_V` 写进同一个 zero-copy mmap-able
`bank.safetensors`；并发写由 `filelock` 串行化，读端凭 `os.replace` 原子
切换只看到完整快照。落盘内容包含 Phase S 的 `LOPIProfile`，重新加载会同
时恢复按架构的校准。格式版本：`ulopi_v36`。

$$
\mathrm{config\_sha} \;=\; \mathrm{sha256}\!\bigl(\,\mathrm{shape}\;\Vert\;\mathrm{LOPIConfig}\;\Vert\;\tau\;\Vert\;\mathrm{shield}\;\Vert\;\mathrm{VScale}\bigr)
$$

* 文件：[`deltamemory/memory/bank_persistence.py`](deltamemory/memory/bank_persistence.py)
* 公开符号：`save_bank`、`load_bank`、`list_banks`、`compute_config_sha`、
  `resolve_location`。
* 往返测试：`tests/test_bank_persistence.py`。

## 阶段表

| Phase | 内容 | 报告目录 | 状态 |
|---|---|---|---|
| Stages 0–14 | v1 → v3（writer / address bank / K-projector） | `reports/cleanroom/{stage13b_*,stage14_*}/`、`transcripts/v31_intervention/` | 已被取代；详见 [`docs/HISTORY.md`](docs/HISTORY.md) |
| Stage 15 / v3.1 | attn-native bank + 按架构 α + 跨架构 adapter | `reports/cleanroom/v31_bench/`、`transcripts/v31_intervention/` | Gemma-4 / Qwen3 在 GB10/Mac 上复现 |
| Stage 16 / v3.2 | mHC 谱 shield（bank 列向 column-cap） | `reports/cleanroom/mhc_flagship_sweep/` | σ_max(W) ≤ 1；α=0 bit-equality 保持 |
| R-3 / v3.3 | Dynamic LOPI ablation（A0–A4，630 cells） | `reports/cleanroom/lopi_v33/` | 已预注册，详见 `AGGREGATE.md` / `FINDINGS.md` |
| R-3.5 / v3.4 | 默认翻转 → `orthogonal=False, gaussian=True, derivative=True` | `reports/cleanroom/lopi_v33/R35_NORM_PROBE.md` | 高 α drift 收敛 + α=1 lift 保留 |
| R-4 / v3.4 | 跨架构 α-safety sweep（Gemma / Qwen3 / GLM-4） | `reports/cleanroom/lopi_v33/R4_xarch/` | 12 cell 全部 α=0 bit-equal |
| R-5.1 / v3.4 | Q3 对抗 chat × LOPI（Gemma-4-E2B） | `reports/cleanroom/lopi_v33/R5_q3/` | 仅 LOPI 配置在 α∈{8,10} 把最易事实拉到 partial implant |
| R-6 / v3.4 | 持久化 AttnNativeBank（safetensors + filelock） | `tests/test_bank_persistence.py` | 同 dtype 下往返 bit-equal |
| **S / v3.5** | U-LOPI 自动校准 profiler（`ulopi_v35`） | `deltamemory/memory/lopi_profiler.py`、`tests/test_lopi_profiler.py`、`tests/test_lopi_universal.py` | 以冷启动 profile 取代固定常数 `norm_base=10.0`；同一份 LOPI 适配 Gemma / Qwen3 / GLM-4 / Llama / GPT-2 |
| **R-7 / v3.6** | bank 侧 V-scale 校准（`ulopi_v36`） | `deltamemory/memory/attn_native_bank.py`、`tests/test_value_scale_calibration.py` | 无 v_norm 家族只 cap M_V RMS、不放大小 V；Gemma 原生 v_norm 不动 |

每阶段长篇叙事日志（rationale、原始 transcript 索引、DeepSeek-32B 边界、
v3.1 图表、按架构 α 默认值）见 [`docs/HISTORY.md`](docs/HISTORY.md)。
每阶段代码 / 配置 diff 见 [`CHANGELOG.md`](CHANGELOG.md)。

## 复现实验

Phase R+ cleanroom 报告所用的 benchmark driver：

```bash
python scripts/run_v31_benchmark.py --help        # v3.1 baseline benchmark
python scripts/run_v31_benchmark_mps.py --help    # Apple Silicon 上的 MPS 变体
```

v3.1 干预 demo（true / 反先验事实，按架构 α 默认值）：

```bash
python scripts/run_intervention_demo.py \
  --model google/gemma-4-E2B \
  --device cuda --dtype bfloat16 \
  --false-facts
```

Apple Silicon 路线见 [`docs/apple_silicon.md`](docs/apple_silicon.md)。
v3.1 反先验测试的原始输入、输出、top-5 预测、目标 log-prob 已逐字提交到
`transcripts/v31_intervention/`；跨架构和 U-LOPI 的数据放在
`reports/cleanroom/lopi_v33/`（R-3、R-3.5、R-4、R-5.1）。

## 测试

```bash
pytest tests/ --ignore=tests/conservation_real_models.py
```

预期：**107 passed, 6 skipped**（共收集 113 个）。被完全忽略的
`conservation_real_models.py` 会下载多 GB HF 权重，opt-in，详见其模块
docstring。Phase S 重点覆盖：
`test_lopi_profiler.py`（profile bit-equality）和
`test_lopi_universal.py`（Gemma / Qwen3 / GLM-4 / Llama / GPT-2 的跨架构
shape + bit-equality）。

## 仓库地图

| 路径 | 用途 |
|---|---|
| `deltamemory/memory/attn_native_bank.py` | AttnNativeBank + 每层 patcher |
| `deltamemory/memory/lopi.py` | Dynamic LOPI v3.4 注入器 |
| `deltamemory/memory/lopi_profiler.py` | U-LOPI Phase S residual profiler |
| `deltamemory/memory/bank_persistence.py` | safetensors + filelock bank 存储 |
| `deltamemory/memory/arch_adapter.py` | 各架构 adapter + α 默认值 |
| `deltamemory/__init__.py` | 顶层公开 API（Phase S） |
| `scripts/run_intervention_demo.py` | 跨架构 true/false-fact 干预 demo |
| `scripts/run_v31_benchmark*.py` | Phase R+ benchmark driver |
| `transcripts/v31_intervention/` | v3.1 原始 input/output/log-prob |
| `reports/cleanroom/` | 预注册和 cleanroom 实验报告 |
| `docs/HISTORY.md` | 长篇阶段叙事日志 |
| `tests/` | 单测 + real-model conservation 检查 |

## 许可证

MIT。见 [`LICENSE`](LICENSE)。
