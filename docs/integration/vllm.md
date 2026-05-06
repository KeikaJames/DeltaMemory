# vLLM integration — supported-injector × backend matrix

> This document tracks which Mneme injectors are supported under which
> inference backend.  See `examples/vllm_integration/README.md` for
> quickstart code.

## Injector × backend compatibility

| Injector | HF transformers | vLLM logits-proc | vLLM custom attn |
|----------|-----------------|------------------|------------------|
| Passthrough (`alpha=0`) | ✅ V.1 | ✅ V.1 | N/A |
| `CAAInjector` (residual-stream) | ✅ existing | 🔜 V.2 | not needed |
| `SCARInjector` | ✅ existing | 🔜 V.3 | not needed |
| `LOPIInjector` | ✅ existing | 🔜 V.3 | not needed |
| `AttnNativeBank` (attention-output) | ✅ existing | 🔒 blocked | 🔒 blocked |

### Notes

- **V.1** (current): `MnemeLogitsProcessor` stub.  Zero delta until a real
  injector is wired.  Alpha=0 is always bit-equal.
- **V.2**: Wire `CAAInjector` steering vector into `set_delta_fn()`.  Requires
  computing a logit-space projection of the residual-stream CAA vector, or
  running the model forward to obtain hidden states (offline vector path is
  simpler and deferred to V.2).
- **V.3**: Extend to `SCARInjector` and `LOPIInjector` using the same
  `set_delta_fn` hook established in V.2.
- **AttnNativeBank**: Requires a custom vLLM attention backend (monkey-patch or
  fork).  Blocked on vLLM internals audit (D-7 in the v0.5 plan).

## Files

```
examples/vllm_integration/
├── mneme_vllm/
│   ├── __init__.py          # exposes MnemeLogitsProcessor
│   ├── logits_processor.py  # core stub (no vLLM import at import time)
│   └── hf_reference.py      # HF greedy-decode adapter for bit-equality tests
├── requirements-vllm.txt    # vllm>=0.6.0,<0.8.0
└── README.md                # quickstart + limitations matrix
tests/
└── test_vllm_logits_processor.py
```
