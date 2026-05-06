"""Entry point: ``uvicorn mneme.api.app:app``."""
from deltamemory.api import app, create_app  # noqa: F401

__all__ = ["app", "create_app"]
