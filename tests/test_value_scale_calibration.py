"""R-7 tests for bank-side V magnitude calibration."""
from __future__ import annotations

import pytest
import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    _scale_bank_value_capture,
)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.float(), ord=2, dim=-1) / (x.size(-1) ** 0.5)


def test_unit_rms_scales_each_head_to_target():
    v = torch.tensor([[[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 8.0]]])
    out = _scale_bank_value_capture(
        v,
        mode="unit_rms",
        has_native_v_norm=True,
        target_rms=0.5,
        eps=1e-6,
    )
    torch.testing.assert_close(_rms(out), torch.full((1, 2), 0.5))


def test_auto_unit_rms_leaves_native_v_norm_unchanged():
    v = torch.randn(2, 3, 8)
    out = _scale_bank_value_capture(
        v,
        mode="auto_unit_rms",
        has_native_v_norm=True,
        target_rms=1.0,
        eps=1e-6,
    )
    torch.testing.assert_close(out, v, rtol=0, atol=0)


def test_auto_unit_rms_scales_no_v_norm_family():
    v = torch.randn(2, 3, 8) * 11.0
    out = _scale_bank_value_capture(
        v,
        mode="auto_unit_rms",
        has_native_v_norm=False,
        target_rms=1.0,
        eps=1e-6,
    )
    torch.testing.assert_close(_rms(out), torch.ones(2, 3), rtol=1e-5, atol=1e-5)


def test_value_scale_none_is_identity():
    v = torch.randn(2, 3, 8)
    out = _scale_bank_value_capture(
        v,
        mode="none",
        has_native_v_norm=False,
        target_rms=1.0,
        eps=1e-6,
    )
    torch.testing.assert_close(out, v, rtol=0, atol=0)


def test_value_scale_rejects_invalid_config():
    v = torch.ones(1, 1, 4)
    with pytest.raises(ValueError, match="value_scale_mode"):
        _scale_bank_value_capture(
            v,
            mode="bad",
            has_native_v_norm=False,
            target_rms=1.0,
            eps=1e-6,
        )
    with pytest.raises(ValueError, match="value_target_rms"):
        _scale_bank_value_capture(
            v,
            mode="unit_rms",
            has_native_v_norm=False,
            target_rms=0.0,
            eps=1e-6,
        )


def test_bank_state_dict_round_trips_value_scale_config():
    bank = AttnNativeBank(num_layers=2, num_kv_heads=1, head_dim=4)
    bank.value_scale_mode = "unit_rms"
    bank.value_target_rms = 0.125
    bank.value_scale_eps = 1e-5
    reloaded = AttnNativeBank.from_state_dict(bank.state_dict())
    assert reloaded.value_scale_mode == "unit_rms"
    assert reloaded.value_target_rms == 0.125
    assert reloaded.value_scale_eps == 1e-5
