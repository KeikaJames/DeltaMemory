"""Tests for Residual Stream Memory."""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from deltamemory.memory.rsm_injector import RSMConfig, RSMInjector, RSMMemoryBank


_VOCAB = 128
_LAYERS = 3
_HIDDEN = 32
_HEADS = 4


@pytest.fixture()
def tiny_gpt2():
    cfg = GPT2Config(
        vocab_size=_VOCAB,
        n_embd=_HIDDEN,
        n_layer=_LAYERS,
        n_head=_HEADS,
        n_positions=64,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


def _ids(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, 12), generator=gen)


def test_capture_shape(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2)
    memory = rsm.capture(_ids())
    assert memory.shape == (_LAYERS, _HIDDEN)
    assert memory.dtype == torch.float32


def test_eta_zero_is_bit_equal_with_nonempty_bank(tiny_gpt2):
    ids = _ids(1)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.0, theta=-1.0))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, _diag = rsm.forward_with_memory(bank, input_ids=ids)
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)


def test_empty_bank_is_bit_equal(tiny_gpt2):
    ids = _ids(2)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.5, theta=-1.0))
    empty = RSMMemoryBank(torch.empty(0, _LAYERS, _HIDDEN), [])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(empty, input_ids=ids)
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 0.0


def test_gate_threshold_suppresses_injection(tiny_gpt2):
    ids = _ids(3)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=1.0, theta=2.0))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(bank, input_ids=ids)
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 0.0


def test_gate_off_changes_output(tiny_gpt2):
    ids = _ids(4)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.25, theta=2.0, gate_off=True))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(bank, input_ids=ids)
        injected = out.logits.detach().clone()

    assert not torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 1.0


def test_gate_off_keeps_similarity_weights_not_all_ones(tiny_gpt2):
    ids = _ids(7)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=1.0, theta=2.0, gate_off=True))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_scores(
            bank,
            torch.tensor([-0.25], dtype=torch.float32),
            input_ids=ids,
        )
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 0.0


def test_shuffled_layers_preserves_shape_and_norms(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2)
    memory = rsm.capture(_ids(5))
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])
    shuffled = bank.shuffled_layers(seed=123)

    assert shuffled.memories.shape == bank.memories.shape
    assert torch.allclose(
        shuffled.memories.norm(dim=-1).sort().values,
        bank.memories.norm(dim=-1).sort().values,
    )


def test_hooks_detach_after_forward(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.2, theta=-1.0))
    ids = _ids(6)
    bank = RSMMemoryBank(rsm.capture(ids).unsqueeze(0), ["fact_0"])
    before = [len(layer._forward_hooks) for layer in rsm.layers]
    rsm.forward_with_memory(bank, input_ids=ids)
    after = [len(layer._forward_hooks) for layer in rsm.layers]
    assert after == before


def test_gate_uniform_uses_all_ones_regardless_of_theta(tiny_gpt2):
    ids = _ids(8)
    # theta is impossibly high so threshold mode would suppress everything,
    # but gate_uniform must still inject all memories with weight 1.
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.5, theta=5.0, gate_mode="uniform"))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_scores(
            bank,
            torch.tensor([-1.0], dtype=torch.float32),
            input_ids=ids,
        )
    assert not torch.equal(base, out.logits)
    assert diag["rsm_activation_rate"] == 1.0
    assert diag["rsm_gate_mode"] == "uniform"


def test_gate_off_back_compat_is_promoted_to_mode_off():
    cfg = RSMConfig(gate_off=True)
    assert cfg.gate_mode == "off"
    cfg2 = RSMConfig(gate_off=True, gate_mode="uniform")
    # Explicit gate_mode wins over legacy flag.
    assert cfg2.gate_mode == "uniform"


def test_score_diagnostics_present_in_diag(tiny_gpt2):
    ids = _ids(9)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.1, theta=-1.0))
    memory = rsm.capture(ids)
    # Two distinct memories: self-similar and a perturbation.
    perturbed = memory + 0.1 * torch.randn_like(memory)
    bank = RSMMemoryBank(torch.stack([memory, perturbed], dim=0), ["a", "b"])
    _out, diag = rsm.forward_with_memory(bank, input_ids=ids)
    for key in (
        "rsm_mean_score",
        "rsm_min_score",
        "rsm_max_score",
        "rsm_score_std",
        "rsm_top_score_minus_mean",
    ):
        assert key in diag
    assert diag["rsm_score_std"] >= 0.0
    assert diag["rsm_top_score_minus_mean"] >= 0.0


def test_pre_block_input_hook_shape_and_eta_zero_bit_equal(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2, RSMConfig(hook_point="pre_block_input"))
    memory = rsm.capture(_ids(10))
    assert memory.shape == (_LAYERS, _HIDDEN)

    ids = _ids(11)
    rsm_zero = RSMInjector(
        tiny_gpt2, RSMConfig(hook_point="pre_block_input", eta=0.0, theta=-1.0)
    )
    mem = rsm_zero.capture(ids)
    bank = RSMMemoryBank(mem.unsqueeze(0), ["fact_0"])
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, _diag = rsm_zero.forward_with_memory(bank, input_ids=ids)
    assert torch.equal(base, out.logits)


def test_pre_block_input_injection_changes_output(tiny_gpt2):
    ids = _ids(12)
    rsm = RSMInjector(
        tiny_gpt2,
        RSMConfig(hook_point="pre_block_input", eta=0.5, theta=-1.0),
    )
    mem = rsm.capture(ids)
    bank = RSMMemoryBank(mem.unsqueeze(0), ["fact_0"])
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(bank, input_ids=ids)
    assert not torch.equal(base, out.logits)
    assert diag["rsm_hook_point"] == "pre_block_input"


def test_mlp_mid_unsupported_on_gpt2_raises(tiny_gpt2):
    # GPT-2 MLP does not expose a `down_proj` attribute, so mlp_mid should
    # surface a clear ValueError instead of silently mis-hooking.
    with pytest.raises(ValueError, match="mlp_mid"):
        RSMInjector(tiny_gpt2, RSMConfig(hook_point="mlp_mid"))


def test_mlp_mid_on_swiglu_like_model():
    """Synthetic SwiGLU-like layer to exercise mlp_mid without large HF models."""
    import torch.nn as nn

    INTERMEDIATE = 24
    HIDDEN = _HIDDEN

    class FakeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
            self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
            self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = FakeMLP()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp(x)

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(_VOCAB, HIDDEN)
            self.layers = nn.ModuleList([FakeLayer() for _ in range(2)])
            self.lm_head = nn.Linear(HIDDEN, _VOCAB, bias=False)

        # Locator hook: lib uses `model.model.layers` or similar — we expose
        # a `.model.layers` chain to satisfy the locator's heuristics.
        def forward(self, input_ids, attention_mask=None, use_cache=False, **kw):
            h = self.embed(input_ids)
            for layer in self.layers:
                h = layer(h)
            logits = self.lm_head(h)
            return type("O", (), {"logits": logits})()

    from deltamemory.memory import _layer_locator as locator
    from deltamemory.memory import rsm_injector as rsm_mod
    model = FakeModel().eval()

    real_loc = locator.get_decoder_layers
    real_rsm = rsm_mod.get_decoder_layers
    locator.get_decoder_layers = lambda m: list(m.layers)
    rsm_mod.get_decoder_layers = lambda m: list(m.layers)
    try:
        rsm = RSMInjector(model, RSMConfig(hook_point="mlp_mid", eta=0.0, theta=-1.0))
        mem = rsm.capture(_ids(13))
        # mlp_mid captures the input to down_proj, which lives in intermediate-dim.
        assert mem.shape == (2, INTERMEDIATE)

        bank = RSMMemoryBank(mem.unsqueeze(0), ["fact_0"])
        with torch.no_grad():
            ids = _ids(14)
            base = model(input_ids=ids).logits.detach().clone()
            out, diag = rsm.forward_with_memory(bank, input_ids=ids)
        assert torch.equal(base, out.logits)  # eta=0 ⇒ bit-equal
        assert diag["rsm_hook_point"] == "mlp_mid"

        # With eta>0 it should change output.
        rsm.config = RSMConfig(hook_point="mlp_mid", eta=0.5, theta=-1.0)
        with torch.no_grad():
            out2, _ = rsm.forward_with_memory(bank, input_ids=ids)
        assert not torch.equal(base, out2.logits)
    finally:
        locator.get_decoder_layers = real_loc
        rsm_mod.get_decoder_layers = real_rsm
