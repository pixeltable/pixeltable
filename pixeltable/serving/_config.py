"""The database configuration a project supplies in its pixeltable.toml."""

from __future__ import annotations

import importlib
import logging
from typing import Any, TypeVar

import pydantic

from pixeltable import config, exceptions as excs

_logger = logging.getLogger(__name__)


def _resolve_module_attr(dotted: str) -> Any:
    """Import a module and resolve an attribute by dotted path.

    For example, 'myapp.queries.search_docs' imports myapp.queries and returns its search_docs attribute.
    """
    split_path = dotted.split(':', 1)
    if len(split_path) != 2:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'invalid query reference {dotted!r}: expected module:attribute'
        )
    module_path, attr_name = split_path
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'could not import module {module_path!r} (from query reference {dotted!r}): {e}',
        ) from e
    if not hasattr(module, attr_name):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'{dotted!r}: module {module_path!r} has no attribute {attr_name!r}'
        )
    return getattr(module, attr_name)


T = TypeVar('T', bound='pydantic.BaseModel')


def _lookup_config(cfg_block: str, name: str, cfg_type: type[T], error_code: excs.ErrorCode) -> T:
    items = config.Config.get().get_value(cfg_block, list)
    if not items:
        raise excs.NotFoundError(error_code, f'No {cfg_block}s found in Pixeltable configuration.')

    cfg = next((c for c in items if c.name == name), None)
    if cfg is None:
        raise excs.NotFoundError(
            error_code,
            f'{cfg_block.title()} {name!r} not found. The following {cfg_block}s are configured:\n'
            f'{", ".join(cfg.name for cfg in items)}',
        )

    assert isinstance(cfg, cfg_type), f'config item {cfg!r} is not of expected type `{cfg_type.__name__}`'
    return cfg


def lookup_database_config() -> config.DatabaseConfig | None:
    """Return the database runtime config from Pixeltable configuration, or None if absent."""
    raw = config.Config.get().get_value('database', dict)
    if raw is None:
        return None
    try:
        return config.DatabaseConfig.model_validate(raw)
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid [pixeltable.database] configuration: {e}'
        ) from e
