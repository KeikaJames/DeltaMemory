"""Compatibility wrapper exposing the production API as mneme.api."""
from deltamemory.api import app, create_app

__all__ = ["app", "create_app"]
