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

_logger = logging.getLogger('pixeltable')


def _resolve_dotted_path(dotted: str) -> Any:
    """Import a module and resolve an attribute by dotted path.

    For example, 'myapp.queries.search_docs' imports myapp.queries and returns its search_docs attribute.
    """
    module_path, _, attr_name = dotted.rpartition('.')
    if not module_path:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'invalid query reference {dotted!r}: expected module.attribute'
        )
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


def _lookup_config(cfg_block: str, name: str, cfg_type: type[T]) -> T:
    items = config.Config.get().get_value(cfg_block, list)
    if not items:
        raise excs.NotFoundError(
            excs.ErrorCode.SERVICE_NOT_FOUND, f'No {cfg_block}s found in Pixeltable configuration.'
        )

    cfg = next((c for c in items if c.name == name), None)
    if cfg is None:
        raise excs.NotFoundError(
            excs.ErrorCode.SERVICE_NOT_FOUND,
            f'{cfg_block.title()} {name!r} not found. The following {cfg_block}s are configured:\n'
            f'{", ".join(cfg.name for cfg in items)}',
        )

    return cfg


def lookup_service_config(name: str) -> config.ServiceConfig:
    """Lookup a ServiceConfig by name from the Pixeltable configuration."""
    return _lookup_config('service', name, config.ServiceConfig)


def lookup_deployment_config(name: str) -> config.DeploymentConfig:
    """Lookup a DeploymentConfig by name from the Pixeltable configuration."""
    return _lookup_config('deployment', name, config.DeploymentConfig)


def create_service_from_config(cfg: config.ServiceConfig) -> 'fastapi.FastAPI':
    """Build a FastAPI instance from a ServiceConfig"""
    Env.get().require_package('fastapi')
    import fastapi

    from pixeltable.serving import FastAPIRouter

    # import user modules so @pxt.query / retrieval_udf definitions are registered
    for mod_path in cfg.modules:
        _logger.info(f'importing module: {mod_path}')
        try:
            importlib.import_module(mod_path)
        except Exception as e:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'could not import module {mod_path!r} listed in `modules`: {e}'
            ) from e

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
                background=route.background,
            )
        elif isinstance(route, config.DeleteRouteConfig):
            t = pxt.get_table(route.table)
            router.add_delete_route(t, path=route.path, match_columns=route.match_columns, background=route.background)
        elif isinstance(route, config.QueryRouteConfig):
            query_fn = _resolve_dotted_path(route.query)
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
