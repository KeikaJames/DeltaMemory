from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model):
        return cls()

    def decode(self, ids):
        return str(ids[0])


class FakePagedAttention(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=4)
        self.paged_attention = FakePagedAttention()

    def forward(self, x):
        return self.paged_attention(x)


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        model = FakeModel()
        runner = SimpleNamespace(model=model)
        worker = SimpleNamespace(model_runner=runner)
        executor = SimpleNamespace(driver_worker=worker)
        self.llm_engine = SimpleNamespace(model_executor=executor)
        self.generated_tensors: list[torch.Tensor] = []

    def generate(self, prompts, sampling_params):
        x = torch.zeros(len(prompts), 1, 4)
        y = self.llm_engine.model_executor.driver_worker.model_runner.model(x)
        self.generated_tensors.append(y.detach())
        return [SimpleNamespace(outputs=[SimpleNamespace(text=f"sum={float(y.sum()):.6f}")])]


class FakePatcher:
    def __init__(self, model):
        self.model = model
        self.patched_entries = 0
        self.inject_entries = []

    def patched(self):
        patcher = self

        class Ctx:
            def __enter__(self):
                patcher.patched_entries += 1

            def __exit__(self, exc_type, exc, tb):
                return False

        return Ctx()

    def injecting(self, bank, alpha=1.0):
        patcher = self

        class Ctx:
            def __enter__(self):
                patcher.inject_entries.append(alpha)

            def __exit__(self, exc_type, exc, tb):
                return False

        return Ctx()


class FakeBank:
    def __init__(self):
        self.M_V = [torch.ones(1, 4)]
        self.empty = False
        self.size = 1


def fake_fresh_bank(_model):
    return FakeBank()


def test_unwrap_vllm_model_supports_0_4_0_5_latest_paths():
    from integrations.vllm.bank_attached_llm import _unwrap_vllm_model

    for path in (
        "llm_engine.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.engine_core.model_executor.driver_worker.model_runner.model",
    ):
        model = FakeModel()
        root = SimpleNamespace()
        obj = root
        parts = path.split(".")
        for part in parts[:-1]:
            child = SimpleNamespace()
            setattr(obj, part, child)
            obj = child
        setattr(obj, parts[-1], model)
        assert _unwrap_vllm_model(root) is model


def test_mock_llm_wires_paged_attention_hooks_and_preserves_alpha_zero():
    from integrations.vllm import BankAttachedLLM

    bllm = BankAttachedLLM(
        "fake-model",
        _llm_cls=FakeLLM,
        _sampling_params_cls=FakeSamplingParams,
        _tokenizer_cls=FakeTokenizer,
        _patcher_cls=FakePatcher,
        _fresh_bank_fn=fake_fresh_bank,
    )

    assert bllm.hook_controller.installed
    assert bllm.hook_controller.module_names == ["paged_attention"]

    before_params = sum(1 for _ in bllm._nn_model.parameters())
    out0 = bllm.generate(["hello"], alpha=0.0)[0].outputs[0].text
    out1 = bllm.generate(["hello"], alpha=1.0)[0].outputs[0].text
    after_params = sum(1 for _ in bllm._nn_model.parameters())

    assert out0 == "sum=0.000000"
    assert out1 != out0
    assert before_params == after_params == 1
    assert bllm.patcher.patched_entries == 1
    assert bllm.patcher.inject_entries == [1.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM GPU e2e requires CUDA")
def test_vllm_tiny_model_generation_changes_with_attached_bank():
    pytest.importorskip("vllm")
    from integrations.vllm import BankAttachedLLM

    model_id = "hf-internal-testing/tiny-random-gpt2"
    bllm = BankAttachedLLM(
        model_id,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=64,
        enforce_eager=True,
        enable_hf_patcher=False,
    )
    bllm.write_facts([("debug", "Debug fact changes the hidden stream.", "Debug")])
    if not bllm.hook_controller.installed:
        pytest.skip("No vLLM paged-attention modules discovered for hook fallback")

    prompt = ["The answer is"]
    plain = bllm.generate(prompt, alpha=0.0, max_new_tokens=4, temperature=0.0)[0].outputs[0].text
    banked = bllm.generate(prompt, alpha=1.0, max_new_tokens=4, temperature=0.0)[0].outputs[0].text

    assert banked != plain
