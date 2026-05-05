FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MNEME_MODEL_ID=Qwen/Qwen2.5-0.5B
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -e . fastapi uvicorn
WORKDIR /app/examples/fastapi_serve
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
