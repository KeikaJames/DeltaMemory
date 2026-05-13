from __future__ import annotations
import asyncio
import inspect
import pytest
def test_generate_with_tiny_random_init_model():
    pytest.importorskip("fastapi"); httpx = pytest.importorskip("httpx")
    from app import create_app
    app = create_app(model_id="tiny-random")
    async def run():
        if "app" in inspect.signature(httpx.AsyncClient).parameters:
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                return await client.post("/generate", json={"prompt":"hi","max_new_tokens":1,"inject":"none"})
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/generate", json={"prompt":"hi","max_new_tokens":1,"inject":"none"})
    r = asyncio.run(run()); assert r.status_code == 200; assert r.json()["model_id"] == "tiny-random"
