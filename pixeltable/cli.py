"""Pixeltable CLI entry point."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixeltable.serving._config import AppConfig, RouteConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog='pxt', description='Pixeltable command-line interface')
    subparsers = parser.add_subparsers(dest='command', required=True)

    serve_parser = subparsers.add_parser('serve', help='Start an HTTP service')
    _add_serve_subparsers(serve_parser)

    args = parser.parse_args()

    if args.command == 'serve':
        _serve(args)
    else:
        parser.print_help()
        sys.exit(1)


def _add_service_args(p: argparse.ArgumentParser) -> None:
    """Add service-level args (host/port/title/prefix) shared across all serve modes."""
    p.add_argument('--host', type=str, default=None, help='Bind address (overrides config default)')
    p.add_argument('--port', type=int, default=None, help='Bind port (overrides config default)')
    p.add_argument('--title', type=str, default=None, help='Service title (overrides config default)')
    p.add_argument('--prefix', type=str, default=None, help='URL prefix (overrides config default)')


def _add_serve_subparsers(serve_parser: argparse.ArgumentParser) -> None:
    serve_sub = serve_parser.add_subparsers(dest='mode', required=True)

    # pxt serve config <path>
    p_config = serve_sub.add_parser('config', help='Load service from a TOML config file')
    p_config.add_argument('config', type=str, help='Path to the service TOML file')
    _add_service_args(p_config)

    # pxt serve insert
    p_insert = serve_sub.add_parser('insert', help='Single insert endpoint')
    p_insert.add_argument('--table', required=True, help='Fully-qualified table path')
    p_insert.add_argument('--path', required=True, help='HTTP path for the endpoint')
    p_insert.add_argument('--inputs', nargs='+', default=None, help='Column names accepted from the JSON body')
    p_insert.add_argument(
        '--uploadfile-inputs',
        nargs='+',
        default=None,
        dest='uploadfile_inputs',
        help='Column names accepted as multipart file uploads',
    )
    p_insert.add_argument('--outputs', nargs='+', default=None, help='Column names returned in the response')
    p_insert.add_argument(
        '--return-fileresponse',
        action='store_true',
        dest='return_fileresponse',
        help='Stream the output as a file response',
    )
    p_insert.add_argument('--background', action='store_true', help='Run the insert in the background')
    _add_service_args(p_insert)

    # pxt serve delete
    p_delete = serve_sub.add_parser('delete', help='Single delete endpoint')
    p_delete.add_argument('--table', required=True, help='Fully-qualified table path')
    p_delete.add_argument('--path', required=True, help='HTTP path for the endpoint')
    p_delete.add_argument(
        '--match-columns',
        nargs='+',
        default=None,
        dest='match_columns',
        help='Column names used to match rows for deletion',
    )
    p_delete.add_argument('--background', action='store_true', help='Run the delete in the background')
    _add_service_args(p_delete)

    # pxt serve query
    p_query = serve_sub.add_parser('query', help='Single query endpoint')
    p_query.add_argument('--query', required=True, help='Dotted Python path to a @pxt.query or retrieval_udf')
    p_query.add_argument('--path', required=True, help='HTTP path for the endpoint')
    p_query.add_argument('--inputs', nargs='+', default=None, help='Parameter names accepted from the request')
    p_query.add_argument(
        '--uploadfile-inputs',
        nargs='+',
        default=None,
        dest='uploadfile_inputs',
        help='Parameter names accepted as multipart file uploads',
    )
    p_query.add_argument('--one-row', action='store_true', dest='one_row', help='Return a single row instead of a list')
    p_query.add_argument(
        '--return-fileresponse',
        action='store_true',
        dest='return_fileresponse',
        help='Stream the output as a file response',
    )
    p_query.add_argument('--background', action='store_true', help='Run the query in the background')
    p_query.add_argument('--method', choices=['get', 'post'], default='post', help='HTTP method (default: post)')
    _add_service_args(p_query)


def _serve(args: argparse.Namespace) -> None:
    from pixeltable.serving._config import AppConfig, ServiceConfig, create_app_from_config, load_app_config

    if args.mode == 'config':
        config = load_app_config(args.config)
    else:
        route = _build_route_from_args(args)
        config = AppConfig(service=ServiceConfig(), routes=[route])

    overrides = {
        k: v
        for k, v in (('host', args.host), ('port', args.port), ('title', args.title), ('prefix', args.prefix))
        if v is not None
    }
    if overrides:
        config = config.model_copy(update={'service': config.service.model_copy(update=overrides)})

    _run(config, create_app_from_config(config))


def _build_route_from_args(args: argparse.Namespace) -> 'RouteConfig':
    from pixeltable.serving._config import DeleteRouteConfig, InsertRouteConfig, QueryRouteConfig

    if args.mode == 'insert':
        return InsertRouteConfig(
            type='insert',
            table=args.table,
            path=args.path,
            inputs=args.inputs,
            uploadfile_inputs=args.uploadfile_inputs,
            outputs=args.outputs,
            return_fileresponse=args.return_fileresponse,
            background=args.background,
        )
    if args.mode == 'delete':
        return DeleteRouteConfig(
            type='delete',
            table=args.table,
            path=args.path,
            match_columns=args.match_columns,
            background=args.background,
        )
    if args.mode == 'query':
        return QueryRouteConfig(
            type='query',
            path=args.path,
            query=args.query,
            inputs=args.inputs,
            uploadfile_inputs=args.uploadfile_inputs,
            one_row=args.one_row,
            return_fileresponse=args.return_fileresponse,
            background=args.background,
            method=args.method,
        )
    raise AssertionError(f'unknown serve mode: {args.mode}')


def _run(config: 'AppConfig', app: Any) -> None:
    import uvicorn

    print(f'Starting Pixeltable service: {config.service.title}')
    print(f'  Listening on http://{config.service.host}:{config.service.port}')
    print(f'  API docs at http://{config.service.host}:{config.service.port}/docs')
    print(f'  Routes: {len(config.routes)}')
    uvicorn.run(app, host=config.service.host, port=config.service.port)
