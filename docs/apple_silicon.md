# Apple Silicon Runtime Notes

RCV-HC needs Hugging Face / PyTorch access to hidden states, attention tensors,
and Gemma Q/K/V projection hooks. For this prototype, the correct local runtime
is PyTorch MPS on Apple Silicon, not Ollama or vLLM.

## Runtime Choice

- Use `--device mps` for real Gemma experiments on Mac.
- Use `--device cpu` only for tests, mock models, or fallback debugging.
- Run MPS experiments outside restricted sandboxes. A restricted shell can make
  `torch.backends.mps.is_available()` return `False` even when the same venv can
  use MPS normally.

## Neural Engine vs MPS

Apple's Neural Engine is mainly exposed through Core ML style compiled-model
execution. RCV-HC currently needs live PyTorch autograd over adapter modules and
runtime hooks inside Gemma attention projections, so the practical backend is
Metal/MPS.

MLX is a plausible future path for Apple Silicon, but this cleanroom prototype
uses Transformers because it already exposes `output_hidden_states`,
`output_attentions`, and module hooks needed by the attention-memory experiment.

## Verified Local Stack

The current working local stack is:

```text
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
transformers==5.7.0
```

Quick check:

```bash
.venv-mac/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
print(torch.ones(1, device="mps"))
PY
```

