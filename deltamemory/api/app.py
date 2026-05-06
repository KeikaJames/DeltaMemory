"""Entry point: ``python -m deltamemory.api`` or ``uvicorn deltamemory.api.app:app``."""
from deltamemory.api import app  # noqa: F401

__all__ = ["app"]
