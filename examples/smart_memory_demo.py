"""Track-M smart-memory demo with synthetic bank tensors.

Run with:
    python examples/smart_memory_demo.py
"""
from __future__ import annotations

from pathlib import Path

import torch

from deltamemory.memory import BankTier, apply_decay, compress_bank, compute_novelty, importance_bias
from deltamemory.memory.attn_native_bank import AttnNativeBank


def _append(bank: AttnNativeBank, vec: torch.Tensor, idx: int) -> None:
    k = [vec.reshape(1, -1)]
    v = [(vec * (idx + 1)).reshape(1, -1)]
    bank.append(k, v, fact_id=f"fact-{idx}", address=f"addr-{idx}")


def main() -> None:
    bank = AttnNativeBank(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32)
    bank.enable_compression = True
    bank.enable_decay = True
    bank.enable_importance = True
    bank.bank_capacity = 3
    bank.compression_threshold = 3

    for i, vec in enumerate(torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.01, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])):
        print(f"novelty before write {i}: {compute_novelty(vec.reshape(1, -1), bank.state_dict()):.3f}")
        _append(bank, vec, i)

    state = bank.state_dict()
    state = compress_bank(state, target_size=3)
    state = apply_decay(state, current_step=bank._x7_global_step + 10, half_life=1000)
    print("compressed rows:", state["M_K"][0].shape[0])
    print("importance bias:", importance_bias(state).tolist())

    cold_path = Path("examples/.smart_memory_demo_cold.safetensors")
    tier = BankTier(state["M_K"][0], state["M_V"][0], cold_path=cold_path)
    warm_idx = tier.demote(("hot", 0))
    cold_idx = tier.demote(warm_idx)
    hot_idx = tier.promote(cold_idx)
    values, indices = tier.query(tier.hot_k[hot_idx[1]], top_k=1)
    print("tier query:", indices, values.shape)
    cold_path.unlink(missing_ok=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
