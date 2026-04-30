"""Model loading and a tiny Gemma-style mock used by cleanroom tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size
        self._id_to_text: dict[int, str] = {0: "<pad>", 1: "<eos>"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.replace("\n", " \n ").split()
        ids = [self._token_id(token) for token in tokens]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids or [self.eos_token_id]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        words = []
        for token_id in ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            words.append(self._id_to_text.get(token_id, f"<tok{token_id}>"))
        return " ".join(words).replace(" \n ", "\n")

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False) -> dict[str, torch.Tensor]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def _token_id(self, token: str) -> int:
        token_id = (abs(hash(token)) % (self.vocab_size - 2)) + 2
        self._id_to_text[token_id] = token
        return token_id


class MockSelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        scale = q.shape[-1] ** -0.5
        scores = q.matmul(k.transpose(-1, -2)) * scale
        causal = torch.triu(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            keep = attention_mask[:, None, :].to(torch.bool)
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        return self.o_proj(attn.matmul(v)), attn[:, None, :, :]


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = MockSelfAttention(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.self_attn(self.input_layernorm(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, attn


class MockInnerModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([MockDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)


class MockGemmaModel(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden_size: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_hidden_layers=num_layers,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
        self.model = MockInnerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        use_cache: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        del use_cache
        hidden = self.model.embed_tokens(input_ids)
        hidden_states = [hidden]
        attentions = []
        for layer in self.model.layers:
            hidden, attn = layer(hidden, attention_mask=attention_mask)
            hidden_states.append(hidden)
            attentions.append(attn)
        hidden = self.model.norm(hidden)
        hidden_states[-1] = hidden
        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
        )


@dataclass
class ModelBundle:
    tokenizer: Any
    model: nn.Module
    model_name: str
    device: torch.device
    dtype: torch.dtype
    family: str
    attentions_available: bool


def load_model_bundle(
    model_name: str,
    device: str = "cpu",
    dtype: str = "float32",
    attn_implementation: str = "eager",
) -> ModelBundle:
    torch_dtype = parse_dtype(dtype)
    actual_device = resolve_device(device)
    if model_name in {"mock-gemma", "mock"}:
        torch.manual_seed(1234)
        tokenizer = SimpleTokenizer()
        model = MockGemmaModel()
        model.to(device=actual_device, dtype=torch_dtype)
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("transformers is required to load non-mock models") from exc
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if attn_implementation != "auto":
            kwargs["attn_implementation"] = attn_implementation
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except Exception as exc:  # pragma: no cover - network/model dependent
            raise RuntimeError(f"failed to load model {model_name!r}: {exc}") from exc
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.output_hidden_states = True
        model.config.output_attentions = True
        model.config.use_cache = False
        model.to(actual_device)
    freeze_model(model)
    attentions_available = probe_attentions(model, actual_device)
    try:
        decoder = get_decoder(model)
        has_layers = hasattr(decoder, "layers")
    except ValueError:
        has_layers = False
    family = "gemma" if "gemma" in model_name.lower() or has_layers else "generic"
    return ModelBundle(tokenizer, model, model_name, actual_device, torch_dtype, family, attentions_available)


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def trainable_base_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def parse_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def get_hidden_size(model: nn.Module) -> int:
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None and hasattr(model.config, "text_config"):
        hidden = getattr(model.config.text_config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config, "n_embd", 0)
    return int(hidden)


def get_vocab_size(model: nn.Module) -> int:
    vocab = getattr(model.config, "vocab_size", None)
    if vocab is None and hasattr(model.config, "text_config"):
        vocab = getattr(model.config.text_config, "vocab_size", None)
    return int(vocab)


def get_decoder(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer
    raise ValueError("unsupported model structure: expected Gemma/Llama-style model.layers, Gemma4 model.language_model.layers, or GPT2 transformer.h")


def exposed_qkv_layers(model: nn.Module) -> list[int]:
    decoder = get_decoder(model)
    layers = getattr(decoder, "layers", getattr(decoder, "h", []))
    result = []
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            result.append(idx)
    return result


def probe_attentions(model: nn.Module, device: torch.device) -> bool:
    try:
        input_ids = torch.ones((1, 3), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_attentions=True, output_hidden_states=True, use_cache=False)
        return out.attentions is not None
    except Exception:
        return False
