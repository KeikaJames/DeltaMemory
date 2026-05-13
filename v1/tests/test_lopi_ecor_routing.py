"""W-T3 round 2: ECOR routing through ``apply_lopi``.

Two redline guarantees (both bit-equal):

1. ``LOPIConfig.use_ecor=False`` (default)               ⇒ identical to legacy.
2. ``LOPIConfig.use_ecor=True`` with ``ECORConfig()``    ⇒ identical to legacy
   (lopi_inject early-returns the additive path).
3. ``LOPIConfig.use_ecor=True`` with ``ECORConfig(enabled=True, soft_blend=0)``
   ⇒ identical to legacy (early-return at ``cfg.soft_blend == 0``).
"""

from __future__ import annotations

import torch

from deltamemory.memory.lopi import (
    LOPIConfig,
    LOPIState,
    apply_lopi,
)
from deltamemory.memory.lopi_inject import ECORConfig


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    B, H, T, D = 2, 4, 3, 8
    out_bank = torch.randn(B, H, T, D)
    v_ctx = torch.randn(B, H, T, D)
    q_post = torch.randn(B, H, T, D)
    return out_bank, v_ctx, q_post


def _legacy(cfg_kwargs: dict, *, layer_idx: int = 2, num_layers: int = 4):
    """Reference: legacy LOPI path with ``use_ecor=False``."""
    out_bank, v_ctx, q_post = _make_inputs(seed=42)
    cfg = LOPIConfig(use_ecor=False, **cfg_kwargs)
    state = LOPIState(num_layers=num_layers)
    return apply_lopi(out_bank, v_ctx, q_post, layer_idx=layer_idx,
                      state=state, cfg=cfg)


def _routed(ecor_cfg: ECORConfig, cfg_kwargs: dict, *,
            layer_idx: int = 2, num_layers: int = 4):
    out_bank, v_ctx, q_post = _make_inputs(seed=42)
    cfg = LOPIConfig(use_ecor=True, ecor_cfg=ecor_cfg, **cfg_kwargs)
    state = LOPIState(num_layers=num_layers)
    return apply_lopi(out_bank, v_ctx, q_post, layer_idx=layer_idx,
                      state=state, cfg=cfg)


def test_use_ecor_false_bit_equal_legacy():
    """The flag default keeps the legacy path."""
    out_bank, v_ctx, q_post = _make_inputs(seed=42)
    state = LOPIState(num_layers=4)
    cfg_default = LOPIConfig(enabled=True, orthogonal=True, gaussian=True,
                             derivative=True)
    # use_ecor must default to False.
    assert cfg_default.use_ecor is False
    expected = apply_lopi(out_bank, v_ctx, q_post, layer_idx=2,
                          state=state, cfg=cfg_default)
    # Re-run via the routed path with disabled ECOR -- result must be identical.
    state2 = LOPIState(num_layers=4)
    cfg_off = LOPIConfig(enabled=True, orthogonal=True, gaussian=True,
                        derivative=True, use_ecor=True,
                        ecor_cfg=ECORConfig(enabled=False))
    out_bank2, v_ctx2, q_post2 = _make_inputs(seed=42)
    actual = apply_lopi(out_bank2, v_ctx2, q_post2, layer_idx=2,
                        state=state2, cfg=cfg_off)
    assert torch.equal(actual, expected), (
        "use_ecor=True with ECORConfig(enabled=False) must be bit-equal "
        "to the legacy path"
    )


def test_use_ecor_soft_blend_zero_bit_equal_legacy():
    """soft_blend=0 ⇒ early-return additive ⇒ bit-equal to legacy."""
    cfg_kwargs = dict(enabled=True, orthogonal=True, gaussian=True,
                      derivative=True)
    expected = _legacy(cfg_kwargs)
    actual = _routed(ECORConfig(enabled=True, soft_blend=0.0), cfg_kwargs)
    assert torch.equal(actual, expected), (
        "use_ecor=True with soft_blend=0 must be bit-equal to legacy"
    )


def test_use_ecor_active_changes_output():
    """Sanity: with soft_blend>0 and a non-zero injection, ECOR must perturb."""
    cfg_kwargs = dict(enabled=True, orthogonal=True, gaussian=True,
                      derivative=True)
    base = _legacy(cfg_kwargs)
    rotated = _routed(ECORConfig(enabled=True, soft_blend=1.0,
                                 max_theta_frac=1.0 / 3.0),
                      cfg_kwargs)
    # The two must differ by more than fp noise on at least some elements.
    diff = (rotated - base).abs().max().item()
    assert diff > 1e-4, (
        f"ECOR active path produced no measurable change vs additive "
        f"(max abs diff = {diff:.2e}); did the routing actually trigger?"
    )


def test_lopi_config_asdict_contains_ecor_keys():
    cfg = LOPIConfig(use_ecor=True,
                     ecor_cfg=ECORConfig(enabled=True, soft_blend=0.5))
    d = cfg.asdict()
    assert d["use_ecor"] is True
    assert isinstance(d["ecor_cfg"], dict)
    assert d["ecor_cfg"]["enabled"] is True
    assert d["ecor_cfg"]["soft_blend"] == 0.5
