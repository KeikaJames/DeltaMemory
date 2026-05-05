# ruff: noqa: E501
from __future__ import annotations
import os
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
InjectMode = Literal["caa", "scar", "lopi", "none"]
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    inject: InjectMode = "none"
class GenerateResponse(BaseModel):
    text: str
    model_id: str
    inject: InjectMode
def create_app(model_id: str | None = None, **_) -> FastAPI:
    app = FastAPI(title="Mneme FastAPI serve")
    app.state.model_id = model_id or os.environ.get("MNEME_MODEL_ID", DEFAULT_MODEL_ID)
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(text=req.prompt, model_id=app.state.model_id, inject=req.inject)
    return app
app = create_app()
