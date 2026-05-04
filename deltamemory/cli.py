"""Command line interface for Mneme experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deltamemory.core.config import MnemeCleanConfig
from deltamemory.engine.attention_memory_engine import AttentionMemoryEngine
from deltamemory.gemma.model_adapter import load_model_bundle
from deltamemory.memory.attention_store import AttentionMemoryStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deltamemory.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--model", default="mock-gemma")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--store", required=True)
    ingest.add_argument("--layers", default="all")
    ingest.add_argument("--block-size", type=int, default=128)
    ingest.add_argument("--memory-dim", type=int, default=128)
    ingest.add_argument("--device", default="cpu")
    ingest.add_argument("--dtype", default="float32")

    ask = sub.add_parser("ask")
    ask.add_argument("--model", default="mock-gemma")
    ask.add_argument("--store", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--answer")
    ask.add_argument("--mode", default="delta_qv")
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument("--alpha-scale", type=float, default=0.2)
    ask.add_argument("--gate-bias", type=float, default=-1.0)
    ask.add_argument("--device", default="cpu")
    ask.add_argument("--dtype", default="float32")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--store", required=True)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "ask":
        return _ask(args)
    if args.command == "inspect":
        return _inspect(args)
    raise AssertionError(args.command)


def _ingest(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    cfg = MnemeCleanConfig(
        model_name=args.model,
        block_size=args.block_size,
        memory_dim=args.memory_dim,
        device=args.device,
        dtype=args.dtype,
        layers=args.layers,
    )
    engine = AttentionMemoryEngine(bundle, cfg)
    summary = engine.ingest(text, layers=args.layers)
    summary["model"] = args.model
    summary["store"] = args.store
    engine.save_store(args.store)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _ask(args: argparse.Namespace) -> int:
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    store = AttentionMemoryStore.load(args.store)
    cfg = MnemeCleanConfig(
        model_name=args.model,
        memory_dim=store.memory_dim,
        top_k=args.top_k,
        alpha_scale=args.alpha_scale,
        gate_bias=args.gate_bias,
        device=args.device,
        dtype=args.dtype,
    )
    engine = AttentionMemoryEngine(bundle, cfg, store=store)
    modes = ["no_memory", "raw_memory", args.mode, f"{args.mode}_zero", f"{args.mode}_random", f"{args.mode}_shuffled", f"{args.mode}_force_gate"]
    # Normalize the common delta_qv controls.
    modes = ["delta_qv_zero" if mode == "delta_qv_zero" else mode for mode in modes]
    modes = ["delta_qv_random" if mode == "delta_qv_random" else mode for mode in modes]
    modes = ["delta_qv_shuffled" if mode == "delta_qv_shuffled" else mode for mode in modes]
    modes = ["delta_qv_force_gate" if mode == "delta_qv_force_gate" else mode for mode in modes]
    result = engine.ask(
        args.question,
        answer=args.answer,
        modes=list(dict.fromkeys(modes)),
        top_k=args.top_k,
        alpha_scale=args.alpha_scale,
        gate_bias=args.gate_bias,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    store = AttentionMemoryStore.load(args.store)
    payload = {
        "memory_count": store.memory_count(),
        "storage_bytes": store.storage_bytes(args.store),
        "metadata": store.metadata,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
