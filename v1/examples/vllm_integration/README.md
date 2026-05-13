# vLLM integration — Mneme delta-memory

## Status

| Milestone | State |
|-----------|-------|
| **V.1** — logits-processor stub + bit-equality test | ✅ complete |
| **V.2** — CAA delta wiring (`CAAInjector`) | 🔜 next |
| **V.3** — SCAR delta wiring (`SCARInjector`) | 🔜 planned |
| AttnNativeBank custom attention backend | 🔒 blocked on vLLM internals |

---

## Quickstart — vLLM

> Requires `pip install -r requirements-vllm.txt` (`vllm>=0.6.0,<0.8.0`).

```python
from mneme_vllm import MnemeLogitsProcessor
from vllm import LLM, SamplingParams

mp = MnemeLogitsProcessor(alpha=1.0)   # delta is zero until V.2 wires CAA

llm = LLM(model="meta-llama/Llama-3.2-1B")
params = SamplingParams(max_tokens=64, logits_processors=[mp])

outputs = llm.generate(["Tell me about memory consolidation."], params)
print(outputs[0].outputs[0].text)
```

Alpha=0 is always bit-equal to unmodified sampling — safe to pass through in
production before CAA vectors are loaded.

---

## Quickstart — HF transformers reference (no vLLM needed)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from mneme_vllm import MnemeLogitsProcessor
from mneme_vllm.hf_reference import hf_run_with_processor

model = AutoModelForCausalLM.from_pretrained("gpt2")
tok   = AutoTokenizer.from_pretrained("gpt2")

mp   = MnemeLogitsProcessor(alpha=0.0)  # passthrough
text = hf_run_with_processor(model, tok, "Hello,", mp, max_new_tokens=8)
print(text)
```

The HF adapter performs manual greedy decoding and calls `processor(token_ids,
logits)` at every step — identical contract to vLLM — enabling a bit-equality
witness across backends.

---

## Limitations matrix

| Feature | Status | Notes |
|---------|--------|-------|
| `MnemeLogitsProcessor` passthrough (`alpha=0`) | ✅ V.1 | bit-equal guarantee |
| Custom delta via `set_delta_fn()` | ✅ V.1 | plug in any `(ids, logits) → Tensor` |
| CAA (`CAAInjector`) delta wiring | 🔜 V.2 | requires hidden-state hook or offline vector |
| SCAR (`SCARInjector`) delta wiring | 🔜 V.3 | depends on V.2 hook infrastructure |
| AttnNativeBank steering | 🔒 blocked | needs custom vLLM attention backend; deferred |

See `docs/integration/vllm.md` for the full injector × backend compatibility matrix.

