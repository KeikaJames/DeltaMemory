# Contributing Guide / 贡献指南

Thank you for your interest in contributing to MnEmE.  
感谢你对 MnEmE 的关注与贡献。

MnEmE is a research prototype for persistent external memory in frozen language models. Contributions should keep the project reproducible, auditable, and aligned with the repository’s safety and responsible-use requirements.  
MnEmE 是一个面向冻结语言模型持久外部记忆的研究原型。所有贡献应保持项目可复现、可审计，并符合仓库中的安全与负责任使用要求。

---

## Project Scope / 项目范围

MnEmE focuses on:  
MnEmE 主要关注：

- External K/V memory for frozen Transformer attention  
  冻结 Transformer attention 中的外部 K/V 记忆
- Attention-native memory banks  
  Attention-native 记忆库
- LOPI / U-LOPI ablations and profiling  
  LOPI / U-LOPI 消融实验与 profiling
- Bank persistence and calibration  
  记忆库持久化与校准
- Reproducible experiments and negative findings  
  可复现实验与负结果记录
- Safety-aware research documentation  
  面向安全的研究文档

Before contributing, please read:  
贡献前请先阅读：

- [`README.md`](README.md)
- [`docs/security.md`](docs/security.md)
- [`docs/HISTORY.md`](docs/HISTORY.md)
- [`CHANGELOG.md`](CHANGELOG.md)

---

## Responsible Use / 负责任使用

This repository involves LLM hidden states, attention-layer mechanisms, tensor banks, and injection mechanisms.  
本仓库涉及 LLM hidden state、attention layer、tensor bank 和 injection 机制。

Contributions must respect the project’s Security Policy and Responsible Use Protocol.  
贡献内容必须遵守项目的安全政策与负责任使用协议。

Please avoid submitting changes that:  
请避免提交以下类型的改动：

- Bypass or weaken safety checks  
  绕过或削弱安全检查
- Hide prompt injection, model manipulation, or unsafe behavior  
  隐藏 prompt injection、模型操控或不安全行为
- Reduce auditability of experiments or outputs  
  降低实验或输出的可审计性
- Add undocumented behavior that changes model outputs  
  添加会改变模型输出但未记录的行为
- Introduce opaque dependencies or external services  
  引入不透明依赖或外部服务

If your contribution affects model behavior, memory injection, persistence, calibration, or evaluation, document the change clearly.  
如果你的贡献会影响模型行为、记忆注入、持久化、校准或评估流程，请清楚记录改动。

---

## Ways to Contribute / 贡献方式

You can contribute by:  
你可以通过以下方式参与：

- Reporting bugs  
  提交 Bug 报告
- Improving documentation  
  改进文档
- Adding tests  
  添加测试
- Fixing reproducibility issues  
  修复可复现性问题
- Improving experiment scripts  
  改进实验脚本
- Adding well-scoped research ablations  
  添加边界清晰的研究消融实验
- Improving type hints, comments, or code clarity  
  改进类型标注、注释或代码可读性
- Reviewing existing experiments and verdict documents  
  审查已有实验和 verdict 文档

---

## Development Environment / 开发环境

MnEmE targets Python 3.11+ and PyTorch-based model workflows.  
MnEmE 面向 Python 3.11+ 与基于 PyTorch 的模型工作流。

Recommended setup:  
推荐环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Then install the project dependencies according to the repository setup.  
然后根据仓库中的实际配置安装项目依赖。

If the project provides an editable install path, use:  
如果项目支持 editable install，可以使用：

```bash
pip install -e .
```

For model-related tests or demos, make sure your environment supports the required backend:  
对于模型相关测试或 demo，请确认环境支持所需后端：

- CUDA
- Apple Silicon MPS
- CPU fallback where applicable

Some tests may download large Hugging Face checkpoints. Read the relevant test docstrings before running them.  
部分测试可能会下载较大的 Hugging Face checkpoint。运行前请先阅读对应测试文件中的说明。

---

## Running Tests / 运行测试

Run the main test suite with:  
使用以下命令运行主要测试：

```bash
pytest tests/ --ignore=tests/conservation_real_models.py
```

The ignored conservation suite may require multi-GB model downloads. Run it only when the change requires real-model conservation checks.  
被忽略的 conservation suite 可能需要下载数 GB 模型。仅在改动确实需要真实模型守恒检查时运行。

For targeted testing, run a specific file:  
如需运行特定测试文件：

```bash
pytest tests/test_attn_native_bank.py
pytest tests/test_bank_persistence.py
pytest tests/test_lopi_profiler.py
pytest tests/test_lopi_universal.py
```

When submitting a PR, include the commands you ran and the result.  
提交 PR 时，请写明你运行过的命令和结果。

---

## Reproducing Experiments / 复现实验

For benchmark or intervention scripts, prefer explicit arguments.  
运行 benchmark 或 intervention 脚本时，建议显式指定参数。

Examples:  
示例：

```bash
python scripts/run_v31_benchmark.py --help
```

```bash
python scripts/run_intervention_demo.py \
  --model google/gemma-4-E2B \
  --device cuda \
  --dtype bfloat16 \
  --false-facts
```

For Apple Silicon, refer to:  
Apple Silicon 环境请参考：

```text
docs/apple_silicon.md
```

Raw experiment outputs should stay local unless they are audited and intentionally promoted into public documentation.  
原始实验输出应保留在本地，经过审计并明确需要公开后再加入文档。

---

## Branch Naming / 分支命名

Use short and descriptive branch names.  
请使用简短且明确的分支名。

Recommended formats:  
推荐格式：

```text
fix/<short-description>
docs/<short-description>
test/<short-description>
refactor/<short-description>
experiment/<short-description>
research/<short-description>
chore/<short-description>
```

Examples:  
示例：

```text
fix/bank-persistence-roundtrip
docs/security-protocol-clarity
test/lopi-profile-bit-equality
experiment/qwen-routing-ablation
```

---

## Commit Messages / Commit 信息

Use clear commit messages. Conventional Commits are recommended.  
请使用清晰的 commit 信息。推荐使用 Conventional Commits。

Common types:  
常见类型：

```text
feat:     add a new feature
fix:      fix a bug
docs:     update documentation
test:     add or update tests
refactor: restructure code without behavior changes
perf:     improve performance
chore:    maintenance changes
```

Examples:  
示例：

```bash
git commit -m "fix: preserve bank config hash across reloads"
git commit -m "test: add bit-equality check for empty bank"
git commit -m "docs: clarify responsible use protocol"
```

---

## Pull Request Guidelines / Pull Request 规范

Before opening a Pull Request, please make sure:  
创建 Pull Request 前，请确认：

- The change is scoped and easy to review  
  改动范围清晰，便于 review
- Tests pass locally  
  本地测试通过
- New behavior is documented  
  新行为已记录
- Research claims include evidence  
  研究结论附有证据
- Experiment results include commands, config, and environment  
  实验结果包含命令、配置和环境信息
- Safety-sensitive changes mention their impact  
  涉及安全的改动说明了影响范围

PRs that affect model behavior should explain:  
影响模型行为的 PR 应说明：

- Which path is affected  
  影响的路径
- Whether `alpha=0` or empty-bank bit-equality is preserved  
  是否保持 `alpha=0` 或空 bank bit-equality
- Whether persistence schema changes  
  持久化 schema 是否变化
- Whether old banks remain loadable  
  旧 bank 是否仍可加载
- Whether new tests cover the change  
  是否有新测试覆盖该改动

---

## Suggested PR Template / 推荐 PR 模板

```markdown
## Summary / 摘要

Describe the change in a few sentences.  
用几句话说明本次改动。

## Motivation / 动机

Explain why this change is needed.  
说明为什么需要此改动。

## Changes / 变更内容

- 
- 
- 

## Tests / 测试

Commands run:  
运行过的命令：

```bash
pytest tests/ --ignore=tests/conservation_real_models.py
```

Result:  
结果：

```text
pass / fail / partial
```

## Experiment Evidence / 实验证据

If applicable, include:  
如适用，请包含：

- Model
- Device
- Dtype
- Dataset or prompt set
- Script command
- Result summary
- Linked verdict or report file

## Safety Impact / 安全影响

Describe any safety, responsible-use, or auditability impact.  
说明对安全、负责任使用或可审计性的影响。

## Compatibility / 兼容性

Mention whether this changes public APIs, saved bank formats, configs, or documented behavior.  
说明是否改变 public API、已保存 bank 格式、配置或文档行为。

## Related Issues / 相关 Issue

Closes #
```

---

## Code Style / 代码风格

Keep the code simple, explicit, and auditable.  
代码应保持简单、明确、可审计。

Guidelines:  
规范：

- Prefer readable code over clever code  
  优先保证可读性
- Keep public APIs stable where possible  
  尽量保持 public API 稳定
- Avoid hidden global state  
  避免隐藏的全局状态
- Document behavior-changing defaults  
  对会改变行为的默认值进行文档说明
- Keep experiment code separate from production paths  
  将实验代码与生产路径保持区分
- Add comments for tensor shape assumptions  
  对 tensor shape 假设添加注释
- Preserve deterministic behavior where practical  
  尽量保持确定性行为

---

## Testing Requirements / 测试要求

Add or update tests when changing:  
以下改动需要添加或更新测试：

- Attention bank behavior  
  attention bank 行为
- LOPI / U-LOPI logic  
  LOPI / U-LOPI 逻辑
- Bank persistence  
  bank 持久化
- Architecture adapters  
  架构适配器
- Value-scale calibration  
  value-scale 校准
- Public APIs  
  public API
- Experiment analysis scripts  
  实验分析脚本

Important test properties:  
重要测试属性：

- Empty-bank behavior  
  空 bank 行为
- `alpha=0` conservation  
  `alpha=0` 守恒
- Shape compatibility across model families  
  不同模型家族的 shape 兼容性
- Save/load round trip  
  保存 / 加载 round trip
- Config hash stability  
  config hash 稳定性
- Reproducibility of reported results  
  报告结果的可复现性

---

## Documentation / 文档

Update documentation when changing:  
以下内容变化时请更新文档：

- Installation or setup  
  安装或环境配置
- Public APIs  
  public API
- Experiment workflow  
  实验流程
- Bank schema or persistence behavior  
  bank schema 或持久化行为
- Safety assumptions  
  安全假设
- Model support  
  模型支持范围
- Known limitations or negative findings  
  已知限制或负结果

Relevant documentation locations:  
相关文档位置：

```text
README.md
README.zh-CN.md
docs/
docs/api/
docs/HISTORY.md
CHANGELOG.md
```

---

## Research Claims / 研究结论

Research claims should be specific and evidence-backed.  
研究结论应具体，并有证据支持。

When adding a claim, include:  
添加结论时，请包含：

- The exact experiment or script  
  精确的实验或脚本
- Model name and version  
  模型名称与版本
- Hardware backend  
  硬件后端
- Dtype  
  数据类型
- Dataset or prompt source  
  数据集或 prompt 来源
- Number of trials or cells  
  trial 或 cell 数量
- Metrics and thresholds  
  指标与阈值
- Failure cases  
  失败案例
- Raw output location if available  
  如有原始输出，请注明位置

Negative results are valuable and should be documented clearly.  
负结果有价值，应清楚记录。

---

## Security-Sensitive Changes / 涉及安全的改动

Changes are security-sensitive if they affect:  
如果改动影响以下内容，则属于安全敏感改动：

- Memory injection behavior  
  记忆注入行为
- Prompt or hidden-state handling  
  prompt 或 hidden-state 处理
- Model output steering  
  模型输出 steering
- External services or network calls  
  外部服务或网络调用
- Persistence of model-derived tensors  
  模型派生 tensor 的持久化
- Loading untrusted files  
  加载不可信文件
- Evaluation or reporting of unsafe behavior  
  不安全行为的评估或报告

For such changes, document the risk and mitigation.  
此类改动请记录风险和缓解方式。

---

## Dependency Policy / 依赖政策

Keep dependencies minimal and justified.  
依赖应保持精简，并说明必要性。

Before adding a dependency, consider:  
添加依赖前，请考虑：

- Why it is needed  
  为什么需要它
- Whether the standard library or existing dependencies can cover it  
  标准库或现有依赖是否已经足够
- Whether it affects installation size  
  是否影响安装体积
- Whether it introduces network access  
  是否引入网络访问
- Whether it affects reproducibility  
  是否影响可复现性
- Whether it is actively maintained  
  是否仍在维护

---

## Compatibility / 兼容性

Avoid breaking compatibility unless the change is intentional and documented.  
请避免破坏兼容性；如确需破坏，请明确记录。

Compatibility-sensitive areas include:  
兼容性敏感区域包括：

- Saved bank format  
  已保存 bank 格式
- Config hashes  
  config hash
- Public imports from `deltamemory`  
  来自 `deltamemory` 的 public import
- Model architecture adapters  
  模型架构适配器
- Default alpha or calibration behavior  
  默认 alpha 或校准行为
- Experiment scripts referenced by documentation  
  文档中引用的实验脚本

---

## Issue Reports / Issue 报告

When reporting a bug, include:  
报告 Bug 时，请包含：

```markdown
## Description / 描述

## Steps to Reproduce / 复现步骤

## Expected Behavior / 期望行为

## Actual Behavior / 实际行为

## Environment / 环境

- OS:
- Python:
- PyTorch:
- Transformers:
- Device:
- Dtype:
- Model:

## Logs / 日志

## Additional Context / 其他信息
```

---

## Feature Requests / 功能建议

When proposing a feature, include:  
提出功能建议时，请包含：

```markdown
## Proposal / 提案

## Use Case / 使用场景

## Expected Behavior / 预期行为

## Research or Engineering Motivation / 研究或工程动机

## Compatibility Impact / 兼容性影响

## Safety Impact / 安全影响

## Possible Implementation / 可能实现
```

---

## Review Expectations / Review 预期

Review focuses on:  
Review 重点包括：

- Correctness  
  正确性
- Reproducibility  
  可复现性
- Safety impact  
  安全影响
- API stability  
  API 稳定性
- Test coverage  
  测试覆盖
- Documentation quality  
  文档质量
- Research evidence  
  研究证据

Please keep discussions technical, specific, and respectful.  
讨论请保持技术性、具体和尊重。

---

## License / 许可证

By contributing to this project, you agree that your contributions are licensed under the project’s MIT License.  
向本项目提交贡献，即表示你同意你的贡献遵循本项目的 MIT License。

See [`LICENSE`](LICENSE).  
详见 [`LICENSE`](LICENSE)。
