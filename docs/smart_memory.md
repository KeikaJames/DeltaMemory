# Track-M Smart Memory Management

Track-M adds optional lifecycle controls for `AttnNativeBank` rows. All flags are off by default, introduce no `nn.Parameter`, and run under `torch.no_grad()`.

## Modules

- `bank_compression.compress_bank(bank_state, target_size)`: greedily merges cosine-similar K rows into centroids and averages V with `merge_counts` utility tracking.
- `bank_decay.apply_decay(bank_state, current_step, half_life=1000)`: applies Ebbinghaus soft decay to V and hard-erases rows below `decay_erase_threshold`.
- `bank_importance.compute_novelty(K_new, bank_state)`: returns `1 - max(cos)`; `importance_bias(bank_state)` returns a `[bank_len]` score multiplier.
- `bank_tiering.BankTier`: manages HOT tensors, WARM CPU tensors, and COLD safetensors via `promote(idx)`, `demote(idx)`, and `query(K_query, top_k)`.

## Enabling

Use runtime flags on `AttnNativePatcher` or the bank:

```python
patcher = AttnNativePatcher(
    model,
    enable_compression=True,
    enable_decay=True,
    enable_importance=True,
    enable_tiering=True,
    compression_threshold=1024,
)
bank = fresh_bank(model)
bank.bank_capacity = 1024
```

Defaults remain off, so the α=0 path still bypasses bank injection (`alpha > 0` is required before bank scores are used).

## Demo

See `examples/smart_memory_demo.py` for an end-to-end synthetic example that exercises compression, decay, importance, and HOT/WARM/COLD tiering together.
