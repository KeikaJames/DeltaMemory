# Mneme FastAPI serving scaffold

`POST /generate` accepts `{prompt, max_new_tokens, inject}` where `inject` is `caa`, `scar`, `lopi`, or `none`. Default model: `Qwen/Qwen2.5-0.5B`; override with `MNEME_MODEL_ID`.

```bash
python -m pip install -e ../.. fastapi uvicorn
uvicorn app:app --host 127.0.0.1 --port 8000
docker build -f examples/fastapi_serve/Dockerfile -t mneme-fastapi .
docker run --rm -p 8000:8000 mneme-fastapi
```

Env vars: `MNEME_MODEL_ID`, `MNEME_DTYPE`, `MNEME_INJECT_LAYER`, `MNEME_CAA_ALPHA`, `MNEME_SCAR_ALPHA`, `MNEME_SCAR_K`, `MNEME_POS_PROMPTS`, `MNEME_NEG_PROMPTS`.

Security: bind `0.0.0.0` only behind a reverse proxy with TLS, auth, request limits, rate limits, and audit logs.
