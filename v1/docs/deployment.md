# Mneme production deployment

## vLLM deep integration

`integrations.vllm.BankAttachedLLM` wraps `vllm.LLM`, unwraps the underlying
`torch.nn.Module` across vLLM 0.4, 0.5, and current executor layouts, and
attaches Mneme's bank without modifying base weights. When `alpha=0` or the bank
is empty, generation calls are passed straight through with no active hook to
preserve bit-equal vLLM logits.

```python
from integrations.vllm import BankAttachedLLM

llm = BankAttachedLLM(
    "/home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it",
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=4096,
)
llm.write_facts([("city_france", "Paris is the capital of France.", "Paris")])
outputs = llm.generate(["The capital of France is"], alpha=1.0, max_new_tokens=16)
```

For vLLM PagedAttention, Mneme installs a forward-hook fallback on the exposed
paged-attention layer. This creates no `nn.Parameter` and keeps the frozen-base
contract. Kernel-level pre-RoPE Q/K/V interception still depends on vLLM
internal APIs, so validate production parity on GB10 with the HF/vLLM cosine
checks before serving traffic.

## Docker

Build the multi-stage CUDA 12.6 production image from the repository root:

```bash
make docker-build
# local single-arch image instead of cache-only validation:
PLATFORMS=linux/amd64 OUTPUT=type=docker make docker-build
```

The Dockerfile supports `linux/amd64` and `linux/arm64` (GB10) via Buildx. The
runtime stage starts `uvicorn mneme.api.app:app` and healthchecks
`GET /health`.

Local compose smoke test:

```bash
docker compose -f docker/docker-compose.dev.yml up --build
curl http://127.0.0.1:8000/health
```
