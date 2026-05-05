#!/usr/bin/env python
"""Throughput / latency / memory harness for Mneme injector comparisons."""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _device(backend: str) -> torch.device:
    if backend == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("backend=cuda requested but CUDA is not available")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("backend=mps requested but MPS is not available")
    return torch.device(backend)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(usage) / (1024 * 1024)
    return float(usage) / 1024


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory()) / (1024 * 1024)
    return 0.0


def _build_smoke_model(dtype: torch.dtype):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(cfg).eval().to(dtype=dtype)


def _load_model(args: argparse.Namespace, dtype: torch.dtype):
    if args.smoke:
        return _build_smoke_model(dtype)
    if not args.model:
        raise SystemExit("--model is required unless --smoke is set")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()


def _decoder_layers(model: Any) -> list[Any]:
    from deltamemory.memory._layer_locator import get_decoder_layers

    return get_decoder_layers(model)


def _model_hidden_size(model: Any) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    return int(cfg.hidden_size)


def _inject_context(model: Any, inject: str, device: torch.device):
    if inject == "none":
        return contextlib.nullcontext()

    layers = _decoder_layers(model)
    layer = max(0, len(layers) // 2)
    hidden = _model_hidden_size(model)

    if inject == "caa":
        from deltamemory.memory.caa_injector import CAAConfig, CAAInjector

        inj = CAAInjector(model, CAAConfig(inject_layer=layer, alpha=1.0), device=device)
        gen = torch.Generator(device="cpu").manual_seed(13)
        inj.steering_vector = torch.randn(hidden, generator=gen, dtype=torch.float32).to(device)
        return inj

    if inject == "scar":
        from deltamemory.memory.scar_injector import SCARInjector

        inj = SCARInjector(model, alpha=1.0, layers=[layer], k=2)
        basis = torch.zeros(hidden, 2, dtype=torch.float32)
        basis[:2, :2] = torch.eye(2)
        inj.basis[layer] = basis
        inj.target_mean[layer] = torch.zeros(hidden, dtype=torch.float32)
        return inj

    raise ValueError(f"unknown inject mode {inject!r}")


@contextlib.contextmanager
def _active_injection(model: Any, inject: str, device: torch.device):
    if inject != "lopi":
        with _inject_context(model, inject, device):
            yield
        return

    from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank
    from deltamemory.memory.lopi import LOPIConfig, LOPIState

    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    gen = torch.Generator(device="cpu").manual_seed(17)
    dtype = next(model.parameters()).dtype
    for i in range(bank.num_layers):
        d = bank.head_dims[i]
        shape = (2, bank.num_kv_heads, d)
        bank.M_K[i] = torch.randn(shape, generator=gen, dtype=dtype).to(device)
        bank.M_V[i] = torch.randn(shape, generator=gen, dtype=dtype).to(device)
    bank.fact_ids = ["bench0", "bench1"]
    bank.lopi_cfg = LOPIConfig(enabled=True, orthogonal=True, gaussian=True, derivative=True)
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)
    with patcher.patched(), patcher.injecting(bank, alpha=1.0):
        yield


def _one_iter(model: Any, input_ids: torch.Tensor, device: torch.device) -> tuple[float, int]:
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        next_ids = torch.zeros(input_ids.size(0), 1, dtype=torch.long, device=device)
        try:
            model(input_ids=next_ids, past_key_values=out.past_key_values, use_cache=True)
        except Exception:
            model(input_ids=next_ids, use_cache=False)
    _sync(device)
    return (time.perf_counter() - start) * 1000.0, input_ids.numel() + input_ids.size(0)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model id or local path; unused with --smoke")
    parser.add_argument("--backend", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--inject", choices=["none", "caa", "scar", "lopi"], default="none")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="Use tiny random-init Qwen3; no downloads")
    args = parser.parse_args(argv)

    device = _device(args.backend)
    dtype = _dtype(args.dtype)
    model = _load_model(args, dtype).to(device).eval()
    vocab = int(getattr(model.config, "vocab_size", 256))
    seq = min(args.seq, int(getattr(model.config, "max_position_embeddings", args.seq)))
    gen = torch.Generator(device="cpu").manual_seed(123)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _active_injection(model, args.inject, device), out_path.open("w", encoding="utf-8") as fh:
        for _ in range(args.warmup):
            ids = torch.randint(0, vocab, (args.batch, seq), generator=gen).to(device)
            _one_iter(model, ids, device)
        for i in range(args.iters):
            ids = torch.randint(0, vocab, (args.batch, seq), generator=gen).to(device)
            latency_ms, tokens = _one_iter(model, ids, device)
            latencies.append(latency_ms)
            rec = {
                "iter": i,
                "model": args.model or "tiny-random-qwen3",
                "backend": device.type,
                "dtype": args.dtype,
                "inject": args.inject,
                "batch": args.batch,
                "seq": seq,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "tokens_per_sec": tokens / (latency_ms / 1000.0),
                "peak_vram_mb": _memory_mb(device),
                "rss_mb": _rss_mb(),
                "smoke": bool(args.smoke),
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    total_tokens = (args.batch * seq + args.batch) * max(1, args.iters)
    total_time = sum(latencies) / 1000.0
    summary = {
        "tokens/sec": total_tokens / total_time if total_time else 0.0,
        "p50_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": _percentile(latencies, 0.95),
        "p99_ms": _percentile(latencies, 0.99),
        "peak_vram_mb": _memory_mb(device),
        "rss_mb": _rss_mb(),
    }
    print("| metric | value |")
    print("|---|---:|")
    for key, value in summary.items():
        print(f"| {key} | {value:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
