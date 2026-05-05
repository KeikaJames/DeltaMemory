from __future__ import annotations

import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from deltamemory.security import (
    AccessGuard,
    AuditLogger,
    BankAuthError,
    Role,
    audit_event,
    tensor_sha256,
)

_VOCAB = 256
_HIDDEN = 64
_LAYERS = 4
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16


@pytest.fixture()
def tiny_qwen3():
    cfg = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval()


class _Tokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=64):
        ids = [min(ord(ch), _VOCAB - 1) for ch in text][:max_length] or [0]
        x = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": x, "attention_mask": torch.ones_like(x)}


def _input_ids(seed: int = 42, n: int = 16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, n), generator=g)


def _required_event_fields() -> set[str]:
    return {
        "ts_ns",
        "event_type",
        "injector",
        "layer",
        "alpha",
        "signal_summary",
        "vector_hash",
        "actor",
        "request_id",
    }


def test_audit_logger_writes_json_lines_with_required_fields(tmp_path):
    path = tmp_path / "audit.jsonl"
    vector = torch.arange(4, dtype=torch.float32)
    with AuditLogger(path=str(path)):
        audit_event(
            event_type="inject",
            injector="unit",
            layer=2,
            alpha=0.5,
            signal_summary={"steer_norm": 1.0, "drift_ratio": 0.1, "gate_mean": 1.0},
            vector_tensor=vector,
            actor="alice",
            request_id="req-1",
        )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert _required_event_fields().issubset(row)
    assert row["event_type"] == "inject"
    assert row["vector_hash"].startswith("sha256:")
    assert set(row["signal_summary"]) == {"steer_norm", "drift_ratio", "gate_mean"}


def test_vector_hash_deterministic_for_same_tensor():
    t = torch.arange(8, dtype=torch.float32)
    assert tensor_sha256(t) == tensor_sha256(t.clone())


def test_vector_hash_differs_for_different_tensor():
    assert tensor_sha256(torch.tensor([1.0])) != tensor_sha256(torch.tensor([2.0]))


def test_alpha_zero_caa_emits_no_audit_and_preserves_contract(tiny_qwen3):
    from deltamemory.memory.caa_injector import CAAConfig, CAAInjector

    model = tiny_qwen3
    ids = _input_ids()
    with torch.no_grad():
        baseline = model(input_ids=ids, use_cache=False).logits.detach().clone()

    inj = CAAInjector(model, CAAConfig(inject_layer=1, alpha=0.0))
    inj.steering_vector = torch.randn(_HIDDEN, generator=torch.Generator().manual_seed(13))
    events: list[dict] = []
    with AuditLogger(sink=events.append), inj, torch.no_grad():
        out = model(input_ids=ids, use_cache=False).logits

    assert events == []
    assert (baseline - out).abs().max().item() == 0.0


def _run_caa_forward(model, events: list[dict]) -> None:
    from deltamemory.memory.caa_injector import CAAConfig, CAAInjector

    inj = CAAInjector(model, CAAConfig(inject_layer=1, alpha=0.75))
    inj.steering_vector = torch.randn(_HIDDEN, generator=torch.Generator().manual_seed(13))
    with AuditLogger(sink=events.append), inj, torch.no_grad():
        model(input_ids=_input_ids(), use_cache=False)


def _run_scar_forward(model, events: list[dict]) -> None:
    from deltamemory.memory.scar_injector import SCARInjector

    inj = SCARInjector(model, alpha=0.5, layers=[1], k=2)
    inj.calibrate(
        ["truth alpha", "honest beta", "correct gamma"],
        ["false alpha", "lying beta", "wrong gamma"],
        _Tokenizer(),
        max_n=3,
    )
    with AuditLogger(sink=events.append), inj, torch.no_grad():
        model(input_ids=_input_ids(), use_cache=False)


def _run_lopi_forward(model, events: list[dict]) -> None:
    from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank
    from deltamemory.memory.lopi import LOPIConfig, LOPIState

    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    for layer in range(bank.num_layers):
        gk = torch.Generator().manual_seed(layer)
        gv = torch.Generator().manual_seed(layer + 100)
        d = bank.head_dims[layer]
        bank.M_K[layer] = torch.randn(2, bank.num_kv_heads, d, generator=gk)
        bank.M_V[layer] = torch.randn(2, bank.num_kv_heads, d, generator=gv)
    bank.fact_ids = ["fact0", "fact1"]
    bank.address_strs = ["addr0", "addr1"]
    bank.lopi_cfg = LOPIConfig(enabled=True, orthogonal=True, gaussian=True, derivative=True)
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)

    with AuditLogger(sink=events.append), patcher.patched(), patcher.injecting(bank, alpha=0.75):
        with torch.no_grad():
            model(input_ids=_input_ids(), use_cache=False)


@pytest.mark.parametrize(
    ("injector", "runner"),
    [("caa", _run_caa_forward), ("scar", _run_scar_forward), ("lopi", _run_lopi_forward)],
)
def test_audit_wired_into_all_injectors_during_forward(tiny_qwen3, injector, runner):
    events: list[dict] = []
    runner(tiny_qwen3, events)
    inject_events = [event for event in events if event["event_type"] == "inject"]
    assert inject_events
    assert {event["injector"] for event in inject_events} == {injector}
    assert all(event["vector_hash"].startswith("sha256:") for event in inject_events)
    assert all(
        set(event["signal_summary"]) == {"steer_norm", "drift_ratio", "gate_mean"}
        for event in inject_events
    )


def test_encrypted_bank_round_trip(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet
    from deltamemory.security import load_encrypted, save_encrypted

    path = tmp_path / "bank.enc"
    key = Fernet.generate_key()
    bank = {
        "layer0": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layer1": torch.ones(2, dtype=torch.float32),
    }
    save_encrypted(bank, str(path), key)
    recovered = load_encrypted(str(path), key)
    assert recovered.keys() == bank.keys()
    for name, tensor in bank.items():
        torch.testing.assert_close(recovered[name], tensor)


def test_encrypted_bank_wrong_key_raises(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet
    from deltamemory.security import load_encrypted, save_encrypted

    path = tmp_path / "bank.enc"
    save_encrypted({"x": torch.tensor([1.0])}, str(path), Fernet.generate_key())
    with pytest.raises(BankAuthError):
        load_encrypted(str(path), Fernet.generate_key())


def test_access_guard_allows_admin_rotate_key():
    AccessGuard.check("rotate_key", Role.ADMIN, actor="root")


def test_access_guard_denies_reader_for_bank_store():
    with pytest.raises(PermissionError):
        AccessGuard.check("bank_store", Role.READER, actor="reader")


def test_access_guard_denial_emits_audit_event_when_attached():
    events: list[dict] = []
    with AuditLogger(sink=events.append), pytest.raises(PermissionError):
        AccessGuard.check("bank_store", Role.READER, actor="reader")
    assert len(events) == 1
    assert events[0]["event_type"] == "access_denied"
    assert events[0]["actor"] == "reader"
    assert events[0]["operation"] == "bank_store"
