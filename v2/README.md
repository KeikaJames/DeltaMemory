# v2 — Attention-Side Latent Bank

v2 是当前主线；`v1/` 只保留历史实验和复现材料。

截至 2026-05-17，v2 的结论已经从“可扩展外部记忆系统”收缩为更精确的研究结论：

- 多槽 soft-attention bank 在 E10/E11/E20C 中被反证：随机 bank、常量 bank、shuffle bank 也能产生大幅 NLL 改善，说明早期增益主要不是按内容检索。
- rank-64 K-projector / residual adapter 是主要有效梯度通道；原始“bank content carries fact identity”的解释不成立。
- Phase-D 的 matched-parameter 消融显示，普通 residual adapter 在同一 split 上比 Q/K LoRA 更强，也足以解释“可训练小模块吸收事实模式”的现象。
- E21 证明了一个更小但真实的能力：单 fact、单 slot、只训练一个 `b` 向量，可以让 frozen LLM 对指定 prompt 输出指定 counterfactual。
- E21b 把 E21 协议移植到多个 decoder family，说明该能力不是 Qwen3 单模型偶然现象。

如果要引用 v2，请优先引用本 README、`verdicts/E10_VERDICT.md`、`verdicts/E11_VERDICT.md`、`verdicts/PHASE_D_LORA_VERDICT.md`、`verdicts/E20C_VERDICT.md`、`verdicts/E21_VERDICT.md`、`verdicts/E21B_CROSSMODEL_VERDICT.md` 和 `scripts/prepublish_audit.py` 的输出。`verdicts/V2_FINAL_VERDICT.md` 是历史长稿，仍含旧计划段落和 `[TBD]`，不能作为投稿主证据。

## 当前可信结论

### 已证伪

1. **512-slot bank 不是可靠的 fact-identity memory。**
   E11 显示 iid Gaussian、单行复制、常量向量、真实 bank K=1 等控制组可以给出同量级甚至更大的 NLL 改善。
   三种 seed 的关键均值：E10 real topK8 Δ=-3.856，random topK8 Δ=-3.628，all-random Δ=-5.196；E11 iid Gaussian Δ=-5.582，single-row replicated Δ=-5.752，constant-vector Δ=-2.957，real-bank K=1 Δ=-5.859。K=0 no-bank 控制 Δ=0.000。

2. **E20b 的大幅 lift 不是 item-specific retrieval。**
   E20C 加入 shuffle / held-out / drift audit 后，发现 lift 是全局 style attractor，而不是事实绑定。

3. **早期 NLL lift 不需要 content bank 才能解释。**
   Phase-D 在同一 Qwen3-4B split、同一层、相近训练预算下比较 matched-parameter adapter：plain residual adapter 三 seed mean Δ=-7.628，而 LoRA-q mean Δ=-0.221、LoRA-qk mean Δ=-0.246。plain adapter 卸载后 NLL 回到 base，说明 lift 来自可卸载小模块本身，不是持久事实检索。

4. **多轮 ponder / pause-write 尚未证明带来额外收益。**
   E15 三个 seed 中，K>2 相对 K=2 的 improvement 都是 0.0000；E14 pause-head 训练也没有过关。

### 已证明

1. **E21 单槽 counterfactual injection 可行。**
   对每个 fact 单独训练一个 `b` 向量，base weights 冻结，projector 冻结，bank 只有一个 slot。Qwen3-4B 上 5/5 facts greedy decode 翻转到目标 counterfactual，cross-prompt truth preservation 19/20。

2. **E21b 跨模型复现。**
   可信结果包括：

   | Family | Model | Layer | Steps | Flips | Cross-prompt preserved |
   |---|---|---:|---:|---:|---:|
   | Qwen3 | Qwen3-4B-Instruct-2507 | 9 | 200 | 5/5 | 19/20 |
   | Qwen3 | Qwen3-1.7B | 18 | 500 | 5/5 | 16/20 |
   | Gemma2 | google/gemma-2-2b | 13 | 500 | 2/2 surviving | 1/2 |
   | Qwen2 | Qwen2.5-0.5B-Instruct | 12 | 500 | 1/1 surviving | 0/0 |
   | Llama | TinyLlama-1.1B-Chat-v1.0 | 14 | 500 | 5/5 | 13/20 |

   注意：Gemma3 和 DeepSeek 当前没有可信通过结果；旧 driver 曾产生过文件名与模型元数据不一致的伪结果，已从工作区删除。必须用修复后的 `--out` 或默认按模型命名输出重跑后，才能新增相关声明。

## 代码入口

```
v2/
├── core/
│   ├── attention_bank.py          # per-layer hidden-state bank + heads
│   ├── qwen3_lpl_patch.py         # Qwen3 attention / pause patch
│   ├── gemma2_bank_patch.py       # Gemma2 attention patch
│   ├── gemma3_bank_patch.py       # Gemma3 attention patch, pending trusted run
│   ├── vanilla_bank_patch.py      # Qwen2 / Llama-style attention patch
│   ├── bank_patch_dispatch.py     # model-class based patch dispatcher
│   ├── kproj.py                   # low-rank residual projector
│   ├── runtime.py                 # multi-round runtime
│   ├── retrieval.py               # top-K bank retrieval helpers
│   └── data_io.py                 # v1 bank blob and split helpers
├── experiments/
│   ├── e01_anticheat_b2/
│   ├── e10_topk_retrieval/
│   ├── e11_noise_robustness/
│   ├── e_phase_d_lora/
│   ├── e20c_adversarial_audit/
│   ├── e21_counterfactual_injection/
│   └── e21b_crossmodel/
├── verdicts/
│   ├── V2_FINAL_VERDICT.md      # historical long draft; not paper-facing
│   ├── E10_VERDICT.md
│   ├── E11_VERDICT.md
│   ├── PHASE_D_LORA_VERDICT.md
│   ├── E20C_VERDICT.md
│   ├── E21_VERDICT.md
│   └── E21B_CROSSMODEL_VERDICT.md
└── methodology/
```

## 复现

### E21 original

```bash
python3 v2/experiments/e21_counterfactual_injection/run.py \
  --device mps \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --bank_layer 9 \
  --steps 200 \
  --lr 5e-3
```

### E21b cross-model

```bash
python3 v2/experiments/e21b_crossmodel/run.py \
  --device mps \
  --model Qwen/Qwen3-1.7B \
  --bank_layer 18 \
  --steps 500 \
  --lr 1e-2

python3 v2/experiments/e21b_crossmodel/run.py \
  --device mps \
  --model google/gemma-2-2b \
  --bank_layer 13 \
  --steps 500 \
  --lr 1e-2

python3 v2/experiments/e21b_crossmodel/run.py \
  --device mps \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --bank_layer 12 \
  --steps 500 \
  --lr 1e-2

python3 v2/experiments/e21b_crossmodel/run.py \
  --device mps \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --bank_layer 14 \
  --steps 500 \
  --lr 1e-2
```

`e21b_crossmodel/run.py` 现在默认写到：

```
v2/experiments/e21b_crossmodel/<model_slug>_L<bank_layer>_<steps>.json
```

也可以显式指定：

```bash
python3 v2/experiments/e21b_crossmodel/run.py ... --out v2/experiments/e21b_crossmodel/my_run.json
```

### Phase-D matched PEFT ablation

```bash
python3 v2/experiments/e_phase_d_lora/run.py \
  --method plain_adapter \
  --device mps \
  --seed 0

python3 v2/experiments/e_phase_d_lora/run.py \
  --method lora_qk \
  --device mps \
  --seed 0
```

Paper-facing Phase-D evidence is the 3-seed grid in `v2/experiments/e_phase_d_lora/`, checked by `v2/scripts/prepublish_audit.py`.

## 已知未收口问题

- Gemma3 patch 已接入 dispatcher，但没有可信通过结果。
- DeepSeek / gpt-oss flagship 还没有本地可信跑通。主要 blocker 是 MoE/custom kernels、CUDA/Triton 依赖、权重下载和显存。
- E14 pause-head training 仍有 placeholder 注释，实验结论应按 FAIL 处理，不要作为可用训练流程。
- E15 `forgetful` mode 不是完整实现；当前可信结论只看 cumulative mode。

## 命名

v2 统一使用 **Attention-Side Latent Bank**。旧称 HNM、hippocampus-style memory、long-term/short-term memory 类比只出现在历史材料中；新的结论应避免继续使用“海马记忆”叙事。
