"""YAML-backed config loader / dumper for Mneme injector and service configs.

Reads a YAML file whose top-level structure is::

    injectors:
      lopi: {alpha: 0.5, ...}
      caa:  {alpha: 0.3, ...}
      ...
    service:
      bind_host: "0.0.0.0"
      ...

Returns a flat ``dict[str, BaseModel]`` keyed by logical name
(``"injectors.lopi"``, ``"service"``, …).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel

from deltamemory.config.schemas import (
    AttnNativeBankConfig,
    BaseInjectorConfig,
    CaaConfig,
    LopiConfig,
    MnemeWriterConfig,
    RomeWriterConfig,
    ScarConfig,
    ServiceConfig,
)

# Mapping from YAML injector key → pydantic schema class
_INJECTOR_SCHEMAS: dict[str, type[BaseInjectorConfig]] = {
    "lopi": LopiConfig,
    "caa": CaaConfig,
    "scar": ScarConfig,
    "attn_native_bank": AttnNativeBankConfig,
    "rome_writer": RomeWriterConfig,
    "mneme_writer": MnemeWriterConfig,
}


def load_config(path: Union[str, Path]) -> dict[str, BaseModel]:
    """Parse *path* as YAML and return validated config objects.

    Each top-level YAML key is dispatched to its schema:

    * ``injectors.<name>`` → the corresponding :class:`~deltamemory.config.schemas.BaseInjectorConfig` subclass.
    * ``service``          → :class:`~deltamemory.config.schemas.ServiceConfig`.

    Unknown keys are passed through as raw ``dict`` values so that callers can
    inspect them without raising an error.

    Parameters
    ----------
    path:
        Filesystem path to a YAML file.

    Returns
    -------
    dict[str, BaseModel]
        Flat mapping from logical key → validated pydantic model.
    """
    path = Path(path)
    raw: dict = yaml.safe_load(path.read_text())

    result: dict[str, BaseModel] = {}

    injectors_raw: dict = raw.get("injectors", {}) or {}
    for name, data in injectors_raw.items():
        key = f"injectors.{name}"
        schema = _INJECTOR_SCHEMAS.get(name)
        if schema is None:
            # Unknown injector — store raw dict wrapped in a generic BaseModel
            # so callers always get a consistent type.
            result[key] = _raw_to_model(data)
        else:
            result[key] = schema.model_validate(data)

    if "service" in raw and raw["service"] is not None:
        result["service"] = ServiceConfig.model_validate(raw["service"])

    return result


def dump_config(cfg: dict[str, BaseModel], path: Union[str, Path]) -> None:
    """Serialise *cfg* (as returned by :func:`load_config`) back to YAML.

    Parameters
    ----------
    cfg:
        Mapping from logical key → pydantic model instance.
    path:
        Destination file path.  Parent directories must exist.
    """
    path = Path(path)

    # Re-assemble the nested YAML structure that load_config expects.
    out: dict = {}
    for key, model in cfg.items():
        if key.startswith("injectors."):
            name = key[len("injectors."):]
            out.setdefault("injectors", {})[name] = model.model_dump()
        else:
            out[key] = model.model_dump()

    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _RawConfig(BaseModel):
    """Opaque wrapper for unknown config sections."""

    model_config = {"extra": "allow"}


def _raw_to_model(data: dict) -> _RawConfig:
    return _RawConfig.model_validate(data or {})
