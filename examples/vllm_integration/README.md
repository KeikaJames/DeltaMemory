# vLLM integration plan (design draft)

Design draft only. Use a vLLM `LogitsProcessor`-style request plugin as control plane, then a custom attention wrapper for SCAR/CAA attention-output steering:

```python
a = attention(q, k, v, kv_cache)
if mneme_state.enabled and layer_id in mneme_state.layers:
    B = mneme_state.basis[layer_id]
    target = mneme_state.target_mean[layer_id]
    a = a + alpha * ((target - a) @ B) @ B.T
return a
```

Prototype CAA first to validate batching, dtype/device placement, and `alpha=0` parity; then add SCAR matrices from `SCARInjector`. LOPI/AttnNativeBank requires a later custom attention backend. Use G2's benchmark harness once a prototype exists.
