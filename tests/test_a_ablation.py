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
