"""TOML-driven configuration for Pixeltable HTTP services."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    import fastapi

import pydantic
import toml

import pixeltable as pxt
import pixeltable.func as func

_logger = logging.getLogger('pixeltable')


# Pydantic models for TOML validation --


class ServiceConfig(pydantic.BaseModel):
    title: str = 'Pixeltable'
    prefix: str = ''
    host: str = '0.0.0.0'
    port: int = 8000


class InsertRouteConfig(pydantic.BaseModel):
    type: Literal['insert']
    table: str
    path: str
    inputs: list[str] | None = None
    uploadfile_inputs: list[str] | None = None
    outputs: list[str] | None = None
    return_fileresponse: bool = False
    background: bool = False


class DeleteRouteConfig(pydantic.BaseModel):
    type: Literal['delete']
    table: str
    path: str
    match_columns: list[str] | None = None
    background: bool = False


class QueryRouteConfig(pydantic.BaseModel):
    type: Literal['query']
    path: str
    query: str  # dotted Python path to a @pxt.query or retrieval_udf
    inputs: list[str] | None = None
    uploadfile_inputs: list[str] | None = None
    one_row: bool = False
    return_fileresponse: bool = False
    background: bool = False
    method: Literal['get', 'post'] = 'post'


RouteConfig = Annotated[InsertRouteConfig | DeleteRouteConfig | QueryRouteConfig, pydantic.Field(discriminator='type')]


class AppConfig(pydantic.BaseModel):
    service: ServiceConfig = ServiceConfig()
    modules: list[str] = []
    routes: list[RouteConfig]


def _resolve_dotted_path(dotted: str) -> Any:
    """Import a module and resolve an attribute by dotted path.

    For example, 'myapp.queries.search_docs' imports myapp.queries and returns its search_docs attribute.
    """
    module_path, _, attr_name = dotted.rpartition('.')
    if not module_path:
        raise pxt.Error(f'invalid query reference {dotted!r}: expected module.attribute')
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise pxt.Error(f'could not import module {module_path!r} (from query reference {dotted!r}): {e}') from e
    if not hasattr(module, attr_name):
        raise pxt.Error(f'{dotted!r}: module {module_path!r} has no attribute {attr_name!r}')
    return getattr(module, attr_name)


def load_app_config(config_path: str) -> AppConfig:
    """Load and validate a TOML service configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = toml.load(f)
    try:
        return AppConfig.model_validate(raw)
    except pydantic.ValidationError as e:
        raise pxt.Error(f'invalid service configuration in {config_path}:\n{e}') from e


def create_app_from_config(config: AppConfig) -> 'fastapi.FastAPI':
    """Build a FastAPI instance from an AppConfig"""
    import fastapi

    from ._fastapi import FastAPIRouter

    # import user modules so @pxt.query / retrieval_udf definitions are registered
    for mod_path in config.modules:
        _logger.info(f'importing module: {mod_path}')
        try:
            importlib.import_module(mod_path)
        except ImportError as e:
            raise pxt.Error(f'could not import module {mod_path!r} listed in `modules`: {e}') from e

    app = fastapi.FastAPI(title=config.service.title)
    router = FastAPIRouter()

    for route in config.routes:
        if isinstance(route, InsertRouteConfig):
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
        elif isinstance(route, DeleteRouteConfig):
            t = pxt.get_table(route.table)
            router.add_delete_route(t, path=route.path, match_columns=route.match_columns, background=route.background)
        elif isinstance(route, QueryRouteConfig):
            query_fn = _resolve_dotted_path(route.query)
            if not isinstance(query_fn, func.QueryTemplateFunction):
                raise pxt.Error(
                    f'query reference {route.query!r} resolved to {type(query_fn).__name__}, '
                    f'expected a @pxt.query or retrieval_udf'
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

    app.include_router(router, prefix=config.service.prefix)
    return app
