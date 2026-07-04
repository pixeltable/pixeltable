"""TOML-driven configuration for Pixeltable HTTP services."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, TypeVar

import pydantic

import pixeltable as pxt
import pixeltable.func as func
from pixeltable import config, exceptions as excs
from pixeltable.env import Env

if TYPE_CHECKING:
    import fastapi

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


def lookup_service_config(name: str) -> config.ServiceConfig:
    """Lookup a ServiceConfig by name from the Pixeltable configuration."""
    return _lookup_config('service', name, config.ServiceConfig, excs.ErrorCode.SERVICE_NOT_FOUND)


def lookup_deployment_config(name: str) -> config.DeploymentConfig:
    """Lookup a DeploymentConfig by name from the Pixeltable configuration."""
    return _lookup_config('deployment', name, config.DeploymentConfig, excs.ErrorCode.DEPLOYMENT_NOT_FOUND)


def lookup_database_runtime_config() -> config.DatabaseRuntimeConfig | None:
    """Return the [database] runtime config from pixeltable.toml, or None if absent."""
    raw = config.Config.get().get_value('database', dict)
    if raw is None:
        return None
    try:
        return config.DatabaseRuntimeConfig.model_validate(raw)
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid [database] section in pixeltable.toml: {e}'
        ) from e


def create_service_from_config(cfg: config.ServiceConfig) -> 'fastapi.FastAPI':
    """Build a FastAPI instance from a ServiceConfig"""
    Env.get().require_package('fastapi')
    import fastapi

    from pixeltable.serving import FastAPIRouter

    app = fastapi.FastAPI(title=cfg.name)
    router = FastAPIRouter()

    for route in cfg.routes:
        if isinstance(route, config.InsertRouteConfig):
            t = pxt.get_table(route.table)
            router.add_insert_route(
                t,
                path=route.path,
                inputs=route.inputs,
                uploadfile_inputs=route.uploadfile_inputs,
                outputs=route.outputs,
                return_fileresponse=route.return_fileresponse,
                export_sql=route.export_sql,
                background=route.background,
            )
        elif isinstance(route, config.UpdateRouteConfig):
            t = pxt.get_table(route.table)
            router.add_update_route(
                t,
                path=route.path,
                inputs=route.inputs,
                outputs=route.outputs,
                return_fileresponse=route.return_fileresponse,
                export_sql=route.export_sql,
                background=route.background,
            )
        elif isinstance(route, config.DeleteRouteConfig):
            t = pxt.get_table(route.table)
            router.add_delete_route(t, path=route.path, match_columns=route.match_columns, background=route.background)
        elif isinstance(route, config.QueryRouteConfig):
            query_fn = _resolve_module_attr(route.query)
            if not isinstance(query_fn, func.QueryTemplateFunction):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'query reference {route.query!r} resolved to {type(query_fn).__name__}, '
                    f'expected a @pxt.query or retrieval_udf',
                )
            router.add_query_route(
                path=route.path,
                query=query_fn,
                inputs=route.inputs,
                uploadfile_inputs=route.uploadfile_inputs,
                one_row=route.one_row,
                return_fileresponse=route.return_fileresponse,
                background=route.background,
                method=route.method,
            )
        _logger.info(f'registered {route.type} route: {route.path}')

    app.include_router(router, prefix=cfg.prefix)
    return app
