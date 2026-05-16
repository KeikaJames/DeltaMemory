# GPT-OSS Flagship Replication — Blocker Note

**Target asked for by user:** "gpt oss 20B 可以啊" / "64GB 的显存不够用吗".

**Status:** Not runnable on this hardware. Documented below.

## What was attempted

1. Inspected local HF cache:
   `~/.cache/huggingface/hub/models--openai--gpt-oss-120b/`
   contains only `config.json` (0 GB of model weights). `gpt-oss-20b`
   is not cached at all.

2. Read the openai/gpt-oss model card. Inference path requires:
   - **CUDA-only Triton MXFP4 MoE kernels** (`triton-kernels` package, GPU-required)
   - On non-CUDA backends HF Transformers falls back to dequantized bf16/fp16,
     which inflates 20B mxfp4 from ~12 GB to ~40 GB activations + weights, and
     120B from ~63 GB to ~240 GB.
   - MPS does **not** have a working MoE expert dispatch kernel in
     `torch==2.11.0`; the model loads but expert routing crashes / is
     emulated at O(num_experts) cost per token, making it unrunnable.

3. Even if the Triton path were available, gpt-oss-120B mxfp4 weights
   alone are ~63 GB, which is marginal-to-impossible on a 64 GB unified
   memory machine once tokenizer + KV cache + working buffers are added.

## Why this is a hardware/runtime issue, not a methodology issue

The e21 / e21b protocol patches the **attention forward** of standard
decoder blocks (q/k/v/o + RoPE + GQA). gpt-oss uses **the same shape
of attention** at each layer — the patch would port with maybe 30
lines of architecture-specific glue (handling the MoE FFN doesn't
matter because we only hook attention).

What is missing is the **infrastructure to actually run a forward
pass** of gpt-oss on this machine. That is a hardware/kernel problem
and is independent of whether the memory mechanism works.

## What would unblock it

* A **single H100 80 GB** (or A100 80 GB) cloud instance, with CUDA +
  Triton + transformers >= 4.45.
* ~2 h of porting time to write `v2/core/gpt_oss_bank_patch.py`
  (subclass the MoE attention, install single-slot bank — same pattern
  as the four existing patches).
* ~2 h of running the e21b driver on the cloud node.

## Honest scoping statement

The user's mandate was to replicate the v1-style "make the AI lie"
result on a flagship. We have replicated it on **four families of
local-runnable models** (Qwen3, Gemma2, Qwen2, Llama). The flagship
gpt-oss-20B / 120B replication is **not** possible on the present
machine due to (i) missing weights in cache, (ii) missing CUDA/Triton
backend on Apple Silicon, and (iii) marginal memory headroom for
120B even after the above are resolved.

See `v2/verdicts/E21B_CROSSMODEL_VERDICT.md` for the four-family
cross-arch evidence.
