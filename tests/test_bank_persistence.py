"""Tests for ``deltamemory.memory.bank_persistence`` (Phase R-6)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank
from deltamemory.memory.bank_persistence import (
    VERSION,
    compute_config_sha,
    list_banks,
    load_bank,
    resolve_location,
    save_bank,
    storage_bytes,
)
from deltamemory.memory.lopi import LOPIConfig


def _make_bank(*, n_facts: int = 3, num_layers: int = 4, num_kv_heads: int = 2,
               head_dim: int = 8, dtype=torch.bfloat16, seed: int = 0) -> AttnNativeBank:
    bank = AttnNativeBank(
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        device="cpu", dtype=dtype,
    )
    g = torch.Generator().manual_seed(seed)
    K = [torch.randn(1, num_kv_heads, head_dim, generator=g, dtype=torch.float32).to(dtype)
         for _ in range(num_layers)]
    V = [torch.randn(1, num_kv_heads, head_dim, generator=g, dtype=torch.float32).to(dtype)
         for _ in range(num_layers)]
    for i in range(n_facts):
        bank.append(
            per_layer_K=K, per_layer_V=V,
            fact_id=f"fact_{i}", address=f"addr_{i}",
        )
    return bank


def _bank_tensors_equal(a: AttnNativeBank, b: AttnNativeBank) -> bool:
    if a.num_layers != b.num_layers:
        return False
    if a.size != b.size:
        return False
    for la, lb in zip(a.M_K, b.M_K):
        if not torch.equal(la.cpu(), lb.cpu()):
            return False
    for la, lb in zip(a.M_V, b.M_V):
        if not torch.equal(la.cpu(), lb.cpu()):
            return False
    return True


def test_round_trip_bit_equal(tmp_path: Path):
    bank = _make_bank(n_facts=5)
    loc = save_bank(bank, tmp_path, model_name="dummy/test-arch")
    assert loc.dir.exists()
    assert (loc.dir / "bank.safetensors").exists()
    assert (loc.dir / "meta.json").exists()

    reloaded = load_bank(loc)
    assert _bank_tensors_equal(bank, reloaded)
    assert reloaded.fact_ids == bank.fact_ids
    assert reloaded.address_strs == bank.address_strs


def test_meta_contents(tmp_path: Path):
    bank = _make_bank(n_facts=2, num_layers=3, head_dim=16)
    loc = save_bank(bank, tmp_path, model_name="some/MoDel-id")
    import json
    meta = json.loads(loc.meta_path.read_text())
    assert meta["version"] == VERSION
    assert meta["n_facts"] == 2
    assert meta["num_layers"] == 3
    assert meta["head_dim"] == 16
    assert meta["fact_ids"] == ["fact_0", "fact_1"]
    assert "saved_at_unix" in meta


def test_config_sha_isolates_distinct_configs(tmp_path: Path):
    """Two banks with identical shape but different LOPI cfgs land in distinct dirs."""
    bank_a = _make_bank(n_facts=1)
    bank_b = _make_bank(n_facts=1)
    bank_a.lopi_cfg = LOPIConfig(enabled=False)
    bank_b.lopi_cfg = LOPIConfig(enabled=True, gaussian=True, derivative=True)

    loc_a = save_bank(bank_a, tmp_path, model_name="m")
    loc_b = save_bank(bank_b, tmp_path, model_name="m")
    assert loc_a.config_sha != loc_b.config_sha
    assert loc_a.dir != loc_b.dir
    assert loc_a.dir.exists() and loc_b.dir.exists()


def test_config_sha_stable_across_runs():
    sha1 = compute_config_sha(
        model_name="x", num_layers=4, num_kv_heads=2, head_dim=8,
        head_dims=None, dtype="bfloat16",
    )
    sha2 = compute_config_sha(
        model_name="x", num_layers=4, num_kv_heads=2, head_dim=8,
        head_dims=None, dtype="bfloat16",
    )
    assert sha1 == sha2


def test_list_banks_finds_saved(tmp_path: Path):
    bank = _make_bank(n_facts=1)
    save_bank(bank, tmp_path, model_name="alpha/m1")
    save_bank(bank, tmp_path, model_name="beta/m2")
    all_locs = list_banks(tmp_path)
    assert len(all_locs) == 2
    only_alpha = list_banks(tmp_path, model_name="alpha/m1")
    assert len(only_alpha) == 1
    assert only_alpha[0].model_safe == "alpha_m1"


def test_storage_bytes_grows_with_facts(tmp_path: Path):
    small = _make_bank(n_facts=1)
    large = _make_bank(n_facts=20)
    loc_s = save_bank(small, tmp_path / "s", model_name="m")
    loc_l = save_bank(large, tmp_path / "l", model_name="m")
    assert storage_bytes(loc_l) > storage_bytes(loc_s)


def test_load_unknown_version_raises(tmp_path: Path):
    bank = _make_bank(n_facts=1)
    loc = save_bank(bank, tmp_path, model_name="m")
    import json
    meta = json.loads(loc.meta_path.read_text())
    meta["version"] = "lopi_v99_future"
    loc.meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema version mismatch"):
        load_bank(loc)


def test_dtype_round_trip_bfloat16(tmp_path: Path):
    bank = _make_bank(n_facts=2, dtype=torch.bfloat16)
    loc = save_bank(bank, tmp_path, model_name="m")
    reloaded = load_bank(loc)
    assert reloaded.M_K[0].dtype == torch.bfloat16


def test_overwrite_same_config_sha(tmp_path: Path):
    bank_v1 = _make_bank(n_facts=2, seed=0)
    bank_v2 = _make_bank(n_facts=5, seed=1)  # same shape, different content
    loc1 = save_bank(bank_v1, tmp_path, model_name="m")
    loc2 = save_bank(bank_v2, tmp_path, model_name="m")
    assert loc1.config_sha == loc2.config_sha
    reloaded = load_bank(loc2)
    assert reloaded.size == 5
    assert _bank_tensors_equal(bank_v2, reloaded)


def test_tuple_location_form(tmp_path: Path):
    bank = _make_bank(n_facts=1)
    loc = save_bank(bank, tmp_path, model_name="m")
    reloaded = load_bank((tmp_path, "m", loc.config_sha))
    assert _bank_tensors_equal(bank, reloaded)


# ---------------------------------------------------------------------------
# Phase S — LOPI profile round-trip


def test_legacy_version_still_loadable(tmp_path: Path):
    """v3.4 banks (lopi_v33) must still load under v3.5 (ulopi_v35)."""
    import json
    bank = _make_bank(n_facts=2)
    loc = save_bank(bank, tmp_path, model_name="m")
    meta = json.loads(loc.meta_path.read_text())
    meta["version"] = "lopi_v33"
    loc.meta_path.write_text(json.dumps(meta))
    reloaded = load_bank(loc)
    assert _bank_tensors_equal(bank, reloaded)


def test_lopi_profile_round_trip(tmp_path: Path):
    from deltamemory.memory.lopi_profiler import LOPIProfile
    bank = _make_bank(n_facts=2, num_layers=4)
    bank.lopi_state.profile = LOPIProfile(
        model_name="dummy/test",
        num_layers=4,
        mu_base=[1.0, 2.0, 3.0, 4.0],
        sigma_base=[0.1, 0.2, 0.3, 0.4],
        mu_arch=2,
        eta_sigma=0.7,
        profile_corpus_sha="cafef00d" * 2,
        n_prompts=10,
        dtype="float32",
    )
    loc = save_bank(bank, tmp_path, model_name="dummy/test")
    reloaded = load_bank(loc)
    assert reloaded.lopi_state.profile is not None
    p = reloaded.lopi_state.profile
    assert p.mu_arch == 2
    assert p.eta_sigma == 0.7
    assert p.mu_base == [1.0, 2.0, 3.0, 4.0]
    assert p.sigma_base == [0.1, 0.2, 0.3, 0.4]
    assert p.profile_corpus_sha == "cafef00d" * 2


def test_profile_corpus_sha_isolates_config_sha(tmp_path: Path):
    """Banks with different profile_corpus_sha must hash to different config_sha."""
    from deltamemory.memory.lopi_profiler import LOPIProfile
    bank_a = _make_bank(n_facts=1)
    bank_b = _make_bank(n_facts=1)
    bank_a.lopi_state.profile = LOPIProfile(
        model_name="m", num_layers=4, mu_base=[1.0]*4, sigma_base=[0.1]*4,
        mu_arch=0, eta_sigma=1.0, profile_corpus_sha="a" * 16,
        n_prompts=5, dtype="float32",
    )
    bank_b.lopi_state.profile = LOPIProfile(
        model_name="m", num_layers=4, mu_base=[1.0]*4, sigma_base=[0.1]*4,
        mu_arch=0, eta_sigma=1.0, profile_corpus_sha="b" * 16,
        n_prompts=5, dtype="float32",
    )
    loc_a = save_bank(bank_a, tmp_path, model_name="m")
    loc_b = save_bank(bank_b, tmp_path, model_name="m")
    assert loc_a.config_sha != loc_b.config_sha
