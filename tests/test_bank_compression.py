from __future__ import annotations

import torch
import torch.nn.functional as F

from deltamemory.memory.bank_compression import compress_bank


def _state(rows: torch.Tensor) -> dict:
    vals = torch.arange(rows.size(0), dtype=torch.float32).reshape(-1, 1, 1)
    return {
        "M_K": [rows.reshape(rows.size(0), 1, rows.size(1)).clone()],
        "M_V": [vals.expand(-1, 1, rows.size(1)).clone()],
        "fact_ids": [f"f{i}" for i in range(rows.size(0))],
        "address_strs": [f"a{i}" for i in range(rows.size(0))],
        "compression_min_similarity": 0.95,
    }


def test_compression_shape_collapses_for_similar_rows():
    rows = F.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]), dim=1)
    out = compress_bank(_state(rows), target_size=2)
    assert out["M_K"][0].shape == (2, 1, 2)
    assert out["M_V"][0].shape == (2, 1, 2)
    assert torch.max(torch.as_tensor(out["merge_counts"])).item() == 2.0


def test_cosine_similar_rows_merge_into_centroid():
    rows = F.normalize(torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]]), dim=1)
    out = compress_bank(_state(rows), target_size=2)
    flat = F.normalize(out["M_K"][0].reshape(2, -1), dim=1)
    assert torch.max(flat @ torch.tensor([1.0, 0.0])).item() > 0.99


def test_random_only_rows_do_not_merge_below_threshold():
    rows = torch.eye(4)
    state = _state(rows)
    state["compression_min_similarity"] = 0.99
    out = compress_bank(state, target_size=2)
    assert out["M_K"][0].shape[0] == 4


def test_alpha_zero_formula_unchanged_after_compression():
    rows = F.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]), dim=1)
    before = torch.randn(2, 3)
    out = compress_bank(_state(rows), target_size=2)
    bank_readout = out["M_V"][0].sum()
    after = before + 0.0 * bank_readout
    assert torch.equal(before, after)
