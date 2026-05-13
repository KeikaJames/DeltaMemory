"""Decoder-layer locator — shared utility for all injectors.

Walks the standard HuggingFace decoder-layer paths and returns the
`nn.ModuleList` of transformer blocks. Single source of truth — used by
``CAAInjector``, ``SCARInjector``, and any future injector that needs to
hook a transformer block.

Author: KeikaJames, 2026-05-06.
"""
from __future__ import annotations

from typing import Any

# Ordered probe paths. Most-specific first (HF wraps language_model under
# model.model for some VLMs); GPT-2 ``transformer.h`` last as the fallback.
_DECODER_PATHS: tuple[str, ...] = (
    "model.model.language_model.layers",
    "model.model.layers",
    "model.language_model.model.layers",
    "model.language_model.layers",
    "language_model.layers",
    "model.layers",
    "transformer.h",
)


def get_decoder_layers(model: Any) -> list[Any]:
    """Return the model's transformer-block list.

    Probes the standard HF decoder paths in order. Returns the first
    non-empty module-list found. Raises ``RuntimeError`` if none match.
    """
    for path in _DECODER_PATHS:
        obj: Any = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    raise RuntimeError(
        "get_decoder_layers: could not locate decoder layers on the model. "
        "Expected one of: " + ", ".join(_DECODER_PATHS)
    )


__all__ = ["get_decoder_layers"]
