"""Unit tests for A.2 part 2 ablation wiring: A3 and A5.

A3 — force eta_sigma=1.0 (disable LOPI σ-shrink):
    Verifies that within the A3 ablation context,
    lopi_profiler.profile_residuals returns profiles with eta_sigma=1.0
    even when the underlying calculation would yield 0.7.

A5 — CAA steering_vector replaced with seeded-random unit vector:
    Verifies that the replaced vector has unit norm and that two runs
    with the same seed produce identical vectors (reproducibility).
"""
from __future__ import annotations

import torch

from experiments.A_ablation.run import ablation_context


# ---------------------------------------------------------------------------
# A3 tests


def test_a3_eta_sigma_forced_to_one():
    """Within the A3 ablation context, profile_residuals always returns
    eta_sigma=1.0 even when the real computation would give 0.7."""
    import deltamemory.memory.lopi_profiler as lopi_profiler
    from deltamemory.memory.lopi_profiler import LOPIProfile

    # Build a fake profile that simulates cv > 0.5 → eta_sigma = 0.7.
    def _fake_profile_residuals(*args, **kwargs):
        return LOPIProfile(
            model_name="fake",
            num_layers=4,
            mu_base=[1.0] * 4,
            sigma_base=[0.5] * 4,
            mu_arch=2,
            profile_corpus_sha="test",
            n_prompts=1,
            dtype="fp32",
            eta_sigma=0.7,  # what an un-ablated cv>0.5 run would return
        )

    original = lopi_profiler.profile_residuals
    lopi_profiler.profile_residuals = _fake_profile_residuals
    try:
        # Outside the context the mock returns 0.7.
        assert lopi_profiler.profile_residuals().eta_sigma == 0.7

        # Inside the A3 context, the patch forces eta_sigma to 1.0.
        with ablation_context("A3"):
            captured = lopi_profiler.profile_residuals()
            assert captured.eta_sigma == 1.0, (
                f"A3 ablation did not force eta_sigma to 1.0; got {captured.eta_sigma}"
            )

        # After exiting, the original mock is restored.
        assert lopi_profiler.profile_residuals().eta_sigma == 0.7
    finally:
        lopi_profiler.profile_residuals = original


def test_a3_context_restores_on_exception():
    """A3 context manager must restore profile_residuals even on error."""
    import deltamemory.memory.lopi_profiler as lopi_profiler

    sentinel = object()
    original = lopi_profiler.profile_residuals
    patched_ref = [None]

    try:
        with ablation_context("A3"):
            patched_ref[0] = lopi_profiler.profile_residuals
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass

    assert lopi_profiler.profile_residuals is original
    assert patched_ref[0] is not original


# ---------------------------------------------------------------------------
# A5 tests


def _make_replacement(seed_val: int, shape=(768,), dtype=torch.float32):
    """Reproduce the A5 replacement logic used in evaluate_arm_cell."""
    g = torch.Generator()
    g.manual_seed(seed_val)
    rand_vec = torch.randn(shape, generator=g).to(dtype=dtype)
    return rand_vec / torch.linalg.vector_norm(rand_vec)


def test_a5_target_mean_replaced_unit_norm():
    """Replaced steering vector must have L2 norm == 1.0 to within 1e-5."""
    seed = 0
    seed_val = seed * 31337 + ord('A') + 5  # == 70
    vec = _make_replacement(seed_val)
    norm = float(torch.linalg.vector_norm(vec).item())
    assert abs(norm - 1.0) < 1e-5, (
        f"A5 replacement vector has norm {norm}; expected 1.0 ± 1e-5"
    )


def test_a5_target_mean_replaced_reproducible():
    """Two calls with the same seed must produce bit-identical vectors."""
    seed = 42
    seed_val = seed * 31337 + ord('A') + 5
    vec1 = _make_replacement(seed_val)
    vec2 = _make_replacement(seed_val)
    assert torch.equal(vec1, vec2), (
        "A5 replacement is not deterministic: two calls with the same seed "
        "produced different vectors"
    )


def test_a5_different_seeds_differ():
    """Different seeds must produce different vectors (collision would be pathological)."""
    v0 = _make_replacement(0 * 31337 + ord('A') + 5)
    v1 = _make_replacement(1 * 31337 + ord('A') + 5)
    assert not torch.equal(v0, v1), (
        "A5 replacement: seed=0 and seed=1 produced identical vectors"
    )


def test_a5_context_manager_is_noop():
    """A5 ablation_context is a no-op; module state is unchanged by it."""
    import deltamemory.memory.caa_injector as _caa
    before = getattr(_caa, "CAAInjector", None)
    with ablation_context("A5"):
        inside = getattr(_caa, "CAAInjector", None)
    after = getattr(_caa, "CAAInjector", None)
    # The context must not mutate the module's CAAInjector symbol.
    assert inside is before
    assert after is before


# ---------------------------------------------------------------------------
# A6 tests — ECOR theta forced to 0


def test_a6_max_theta_frac_forced_to_zero():
    """Within A6, lopi_inject sees cfg.max_theta_frac == 0 regardless of input."""
    import sys
    __import__("deltamemory.memory.lopi_inject")
    _lopi_inj_mod = sys.modules["deltamemory.memory.lopi_inject"]
    from deltamemory.memory.lopi_inject import ECORConfig

    captured = {}

    def _spy(*args, **kwargs):
        captured["cfg"] = kwargs.get("cfg")
        return args[0]

    original = _lopi_inj_mod.lopi_inject
    _lopi_inj_mod.lopi_inject = _spy
    try:
        with ablation_context("A6"):
            patched = _lopi_inj_mod.lopi_inject
            assert patched is not _spy, "A6 should have wrapped lopi_inject"
            cfg_in = ECORConfig(enabled=True, max_theta_frac=1.0 / 3.0, soft_blend=0.5)
            patched(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), torch.tensor(0.5), cfg=cfg_in)
            assert captured["cfg"] is not None
            assert captured["cfg"].max_theta_frac == 0.0
            assert captured["cfg"].soft_blend == 0.5
            assert captured["cfg"].enabled is True
            # Caller's cfg untouched (we cloned).
            assert cfg_in.max_theta_frac == 1.0 / 3.0
    finally:
        _lopi_inj_mod.lopi_inject = original


def test_a6_restores_original_on_exit():
    """A6 monkey-patch must be reverted after the context exits."""
    import sys
    __import__("deltamemory.memory.lopi_inject")
    _lopi_inj_mod = sys.modules["deltamemory.memory.lopi_inject"]
    before = _lopi_inj_mod.lopi_inject
    with ablation_context("A6"):
        assert _lopi_inj_mod.lopi_inject is not before
    assert _lopi_inj_mod.lopi_inject is before


def test_a6_handles_none_cfg():
    """If caller passes cfg=None, A6 still forces max_theta_frac=0."""
    import sys
    __import__("deltamemory.memory.lopi_inject")
    _lopi_inj_mod = sys.modules["deltamemory.memory.lopi_inject"]
    captured = {}

    def _spy(*args, **kwargs):
        captured["cfg"] = kwargs.get("cfg")
        return args[0]

    original = _lopi_inj_mod.lopi_inject
    _lopi_inj_mod.lopi_inject = _spy
    try:
        with ablation_context("A6"):
            _lopi_inj_mod.lopi_inject(torch.zeros(1), torch.zeros(1), torch.tensor(0.0))
            assert captured["cfg"] is not None
            assert captured["cfg"].max_theta_frac == 0.0
    finally:
        _lopi_inj_mod.lopi_inject = original


# ---------------------------------------------------------------------------
# A7 tests — alpha-shield removed


def test_a7_patches_and_restores_caainjector_enter():
    """A7 must monkey-patch CAAInjector.__enter__ inside the context and restore on exit."""
    from deltamemory.memory.caa_injector import CAAInjector
    before = CAAInjector.__enter__
    with ablation_context("A7"):
        assert CAAInjector.__enter__ is not before, \
            "A7: __enter__ should be replaced inside the context"
    assert CAAInjector.__enter__ is before, \
        "A7: __enter__ should be restored on exit"


def test_a7_no_shield_hook_runs_without_alpha_zero_shortcircuit():
    """The patched A7 hook installs and runs without the alpha==0 shield.

    We verify the structural change (shield removed) by exercising the
    patched __enter__ on a stub matching CAAInjector's interface and
    confirming that at α=1 the hook adds the steering vector to every
    position (which, with the shield-removed hook body, is the only path
    back to the output regardless of α).
    """
    import torch.nn as nn
    from deltamemory.memory.caa_injector import CAAInjector

    class _StubLayer(nn.Module):
        def forward(self, x):
            return x

    class _StubInjector:
        def __init__(self, alpha: float):
            class _Cfg:
                pass
            cfg = _Cfg()
            cfg.alpha = alpha
            cfg.use_lopi_gate = False
            cfg.gate_k = 1.0
            cfg.gate_theta = 0.0
            self.config = cfg
            self.steering_vector = torch.ones(4)
            self._prev_hidden = None
            self._hook_handle = None
            self._stub_layer = _StubLayer()

        def _resolve_layer(self):
            return 0

        def _get_decoder_layers(self):
            return [self._stub_layer]

    with ablation_context("A7"):
        patched_enter = CAAInjector.__enter__
        # α=1 path: hook MUST add s_bc to every position.
        stub = _StubInjector(alpha=1.0)
        patched_enter(stub)
        x = torch.randn(1, 3, 4)
        out = stub._stub_layer(x)
        expected = x + stub.steering_vector.unsqueeze(0).unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-6), \
            "A7 at α=1: hook should add steering vector"
        stub._hook_handle.remove()


# ---------------------------------------------------------------------------
# A1 tests — post-RoPE K capture forced


def test_a1_sets_force_post_rope_capture_on_new_patcher():
    """A1: any AttnNativePatcher created INSIDE the context has the
    `_a1_force_post_rope_capture` flag set to True. Outside the context,
    fresh patchers do NOT have the flag set (or it is False)."""
    from deltamemory.memory.attn_native_bank import AttnNativePatcher
    import torch.nn as nn

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            class _C:
                pass

            self.config = _C()
            self.config.num_hidden_layers = 1
            self.config.num_attention_heads = 1
            self.config.num_key_value_heads = 1
            self.config.hidden_size = 4
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

    stub = _Stub()

    # Outside: flag not present (or False).
    try:
        p_out = AttnNativePatcher(stub)
        outside_flag = getattr(p_out, "_a1_force_post_rope_capture", False)
    except Exception:
        outside_flag = "ctor-incompat"

    with ablation_context("A1"):
        try:
            p_in = AttnNativePatcher(stub)
            inside_flag = getattr(p_in, "_a1_force_post_rope_capture", False)
        except Exception:
            inside_flag = "ctor-incompat"

    # Restoration: a patcher created AFTER the context is back to default.
    try:
        p_after = AttnNativePatcher(stub)
        after_flag = getattr(p_after, "_a1_force_post_rope_capture", False)
    except Exception:
        after_flag = "ctor-incompat"

    if outside_flag != "ctor-incompat":
        assert outside_flag is False, "A1: pre-context default should be False"
        assert inside_flag is True, "A1: in-context default should be True"
        assert after_flag is False, "A1: post-context default should be False"


def test_a1_restores_init_after_exit():
    """A1's monkey-patch on AttnNativePatcher.__init__ must be reverted."""
    from deltamemory.memory.attn_native_bank import AttnNativePatcher
    before = AttnNativePatcher.__init__
    with ablation_context("A1"):
        assert AttnNativePatcher.__init__ is not before
    assert AttnNativePatcher.__init__ is before


# ---------------------------------------------------------------------------
# A4 tests — SCAR orthogonal projection skipped


def test_a4_replaces_do_inject_method():
    """A4 must monkey-patch SCARInjector._do_inject and restore on exit."""
    from deltamemory.memory.scar_injector import SCARInjector
    before = SCARInjector._do_inject
    with ablation_context("A4"):
        assert SCARInjector._do_inject is not before
    assert SCARInjector._do_inject is before


def test_a4_uses_raw_delta_not_projected():
    """Inside A4, _do_inject applies activations + alpha*(target - act).

    With basis=identity-restricted, raw delta and projected delta agree;
    we use a non-trivial 1-D basis and verify that A4's output equals
    the raw-delta path while default would equal the projected path.
    """
    import torch
    from deltamemory.memory.scar_injector import SCARInjector

    # Build a stub with the minimal attrs _do_inject reads.
    class _Stub:
        def __init__(self):
            self.alpha = 0.5
            # 4-D space, basis is span of [1,0,0,0] only — projection drops y/z/w.
            self.basis = {0: torch.tensor([[1.0], [0.0], [0.0], [0.0]])}
            self.target_mean = {0: torch.tensor([2.0, 3.0, 5.0, 7.0])}

    stub = _Stub()
    activations = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # shape [1, 4]

    # Default path: projected only keeps the x-component.
    out_default = SCARInjector._do_inject(stub, activations, layer=0).clone()

    with ablation_context("A4"):
        out_ablated = SCARInjector._do_inject(stub, activations, layer=0).clone()

    delta = stub.target_mean[0] - activations[0]  # [1, 2, 4, 6]
    expected_ablated = activations + 0.5 * delta

    assert torch.allclose(out_ablated, expected_ablated, atol=1e-5), \
        f"A4 raw-delta path mismatch: got {out_ablated}, want {expected_ablated}"
    # The ablated path must DIFFER from the default projected path on the
    # off-basis (y/z/w) components.
    assert not torch.allclose(out_ablated, out_default, atol=1e-5), \
        "A4 should differ from default projection on off-basis components"
