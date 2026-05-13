"""Mneme production FastAPI application (mneme.api.app).

This module provides the full production API for serving AttnNativeBank-
augmented inference.  It is the intended target of ``scripts/deploy_gb10_e2e.sh``.

Endpoints
---------
GET  /health                — liveness probe
GET  /bank/status           — bank fact count, model info
POST /bank/write            — write one or more facts to the bank
DELETE /bank               — clear the bank
POST /generate              — generate text with (optional) bank injection

Environment variables
---------------------
MNEME_MODEL_PATH            — path to HF model directory (required for real mode)
MNEME_DEVICE                — cuda / mps / cpu  (default: cuda if available)
MNEME_DTYPE                 — bfloat16 / float16 / float32  (default: bfloat16)
MNEME_ALPHA                 — default injection scale  (default: 1.0)
MNEME_MAX_NEW_TOKENS        — default max generation tokens  (default: 64)
MNEME_STUB_MODE             — if "1" / "true", skip model load (for unit tests)

FastAPI / pydantic are optional runtime dependencies.  The module imports
cleanly even when they are absent; ``create_app()`` raises ``ImportError``
with install instructions if called without them.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel as _BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    _BaseModel = object  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

_MODEL_PATH: Optional[str] = os.environ.get("MNEME_MODEL_PATH")
_DEVICE: str = os.environ.get(
    "MNEME_DEVICE",
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
)
_DTYPE_STR: str = os.environ.get("MNEME_DTYPE", "bfloat16")
_DTYPE: torch.dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}.get(_DTYPE_STR, torch.bfloat16)
_ALPHA: float = float(os.environ.get("MNEME_ALPHA", "1.0"))
_MAX_NEW_TOKENS: int = int(os.environ.get("MNEME_MAX_NEW_TOKENS", "64"))
_STUB_MODE: bool = os.environ.get("MNEME_STUB_MODE", "0").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Application state (populated in lifespan)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "patcher": None,
    "bank": None,
    "model_path": _MODEL_PATH,
    "device": _DEVICE,
    "dtype": _DTYPE_STR,
    "loaded_at": None,
}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """Load model + bank at startup; clean up on shutdown."""
    if not _STUB_MODE and _MODEL_PATH:
        _load_model(_MODEL_PATH)
    yield


@asynccontextmanager
async def _null_lifespan(app):
    yield


def _load_model(model_path: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_DTYPE,
        device_map=_DEVICE,
    )
    model.eval()
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    _state["model"] = model
    _state["tokenizer"] = tok
    _state["patcher"] = patcher
    _state["bank"] = bank
    _state["model_path"] = model_path
    _state["loaded_at"] = time.time()


# ---------------------------------------------------------------------------
# Request / response schemas (only defined when FastAPI is available)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class FactItem(_BaseModel):
        fact_id: str
        write_prompt: str
        address: str
        policy: str = "period"

    class WriteBankRequest(_BaseModel):
        facts: list[FactItem]

    class WriteBankResponse(_BaseModel):
        written: int
        total_facts: int

    class BankStatusResponse(_BaseModel):
        model_path: Optional[str]
        num_facts: int
        empty: bool
        device: str
        dtype: str
        stub_mode: bool

    class GenerateRequest(_BaseModel):
        prompt: str
        max_new_tokens: int = _MAX_NEW_TOKENS
        alpha: float = _ALPHA
        temperature: float = 0.0

    class GenerateResponse(_BaseModel):
        text: str
        prompt: str
        alpha: float
        latency_ms: float
        recall_top5: list[str]

    class HealthResponse(_BaseModel):
        status: str
        model_loaded: bool
        vllm_version: Optional[str] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model_path: Optional[str] = None,
    stub_mode: Optional[bool] = None,
    **_,
) -> "FastAPI":
    """Create and return the FastAPI application.

    Parameters
    ----------
    model_path:
        Override ``MNEME_MODEL_PATH`` env var.
    stub_mode:
        Override ``MNEME_STUB_MODE`` env var (useful for unit tests).
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed.  Install it with:\n"
            "    pip install fastapi uvicorn\n"
            "then retry."
        )

    _effective_stub = stub_mode if stub_mode is not None else _STUB_MODE
    if model_path:
        _state["model_path"] = model_path

    _app = FastAPI(
        title="Mneme API",
        description="AttnNativeBank-augmented LLM inference",
        version="0.6.0",
        lifespan=lifespan if not _effective_stub else _null_lifespan,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @_app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=_state["model"] is not None,
        )

    @_app.get("/bank/status", response_model=BankStatusResponse)
    async def bank_status():
        bank = _state.get("bank")
        return BankStatusResponse(
            model_path=_state.get("model_path"),
            num_facts=bank.num_facts if bank and not bank.empty else 0,
            empty=bank.empty if bank else True,
            device=_state["device"],
            dtype=_state["dtype"],
            stub_mode=_effective_stub,
        )

    @_app.post("/bank/write", response_model=WriteBankResponse)
    async def write_bank(req: WriteBankRequest):
        if _effective_stub:
            return WriteBankResponse(written=len(req.facts), total_facts=len(req.facts))
        patcher = _state.get("patcher")
        bank = _state.get("bank")
        tok = _state.get("tokenizer")
        if patcher is None or bank is None or tok is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        from deltamemory.memory.attn_native_bank import write_fact
        for item in req.facts:
            write_fact(
                patcher, bank, tok,
                write_prompt=item.write_prompt,
                fact_id=item.fact_id,
                address=item.address,
                policy=item.policy,
            )
        return WriteBankResponse(
            written=len(req.facts),
            total_facts=bank.num_facts,
        )

    @_app.delete("/bank")
    async def clear_bank():
        if _effective_stub:
            return {"cleared": True}
        model = _state.get("model")
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        from deltamemory.memory.attn_native_bank import fresh_bank
        _state["bank"] = fresh_bank(model)
        return {"cleared": True, "total_facts": 0}

    @_app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        if _effective_stub:
            return GenerateResponse(
                text=req.prompt,
                prompt=req.prompt,
                alpha=req.alpha,
                latency_ms=0.0,
                recall_top5=[],
            )
        patcher = _state.get("patcher")
        bank = _state.get("bank")
        model = _state.get("model")
        tok = _state.get("tokenizer")
        if model is None or tok is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        from deltamemory.memory.attn_native_bank import forward_with_bank

        t0 = time.perf_counter()

        # Top-5 recall probe (fast — single forward pass, no decode loop)
        logits = forward_with_bank(
            patcher, bank, tok, req.prompt, alpha=req.alpha
        )
        top5_ids = logits.topk(5).indices.tolist()
        top5_tokens = [tok.decode([tid]).strip() for tid in top5_ids]

        # Full generation (greedy)
        enc = tok(req.prompt, return_tensors="pt").to(model.device)
        with patcher.patched(), patcher.injecting(bank, alpha=req.alpha), torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else None,
                use_cache=True,
            )
        prompt_len = enc["input_ids"].shape[1]
        new_ids = gen_ids[0, prompt_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return GenerateResponse(
            text=text,
            prompt=req.prompt,
            alpha=req.alpha,
            latency_ms=latency_ms,
            recall_top5=top5_tokens,
        )

    return _app


# ---------------------------------------------------------------------------
# Module-level ``app`` — only instantiated when FastAPI is available.
# For ``uvicorn deltamemory.api.app:app`` usage on GB10.
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    app = create_app()
