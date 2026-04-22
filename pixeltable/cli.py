"""Pixeltable CLI entry point."""

from __future__ import annotations

import argparse
import errno
import json as json_mod
import sys
from typing import TYPE_CHECKING, Any

import pydantic

import pixeltable as pxt
from pixeltable import exceptions as excs

if TYPE_CHECKING:
    from pixeltable.serving._config import AppConfig, RouteConfig


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that appends the epilog examples to stderr on error."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        sys.stderr.write(f'\npxt: error: {message}\n')
        if self.epilog is not None:
            sys.stderr.write(f'\n{self.epilog}\n')
        sys.exit(2)


# Examples mirror the "Quickstart (single-endpoint CLI)" section of
# docs/release/howto/deployment/serving.mdx. Keep in sync if examples change.
_EPILOG_CONFIG = """\
Examples:
  pxt serve config service.toml
  pxt serve config service.toml --port 9000"""

_EPILOG_INSERT = """\
Examples:
  pxt serve insert --table my_dir.my_table --path /generate \\
    --inputs prompt --outputs prompt result --port 8000"""

_EPILOG_DELETE = """\
Examples:
  pxt serve delete --table my_dir.my_table --path /delete"""

_EPILOG_QUERY = """\
Examples:
  pxt serve query --query myapp.queries.search_docs --path /search"""


def main() -> None:
    parser = _Parser(
        prog='pxt',
        description='Pixeltable command-line interface',
        # formatter: make sure we don't remove indentation or other intentional white space
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'pxt {pxt.__version__}')
    subparsers = parser.add_subparsers(dest='command', required=False)

    serve_parser = subparsers.add_parser('serve', help='Start an HTTP service')
    _add_serve_subparsers(serve_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'serve':
            _serve(args)
    except pxt.Error as e:
        _emit_error(str(e), args.json)
        sys.exit(1)


def _emit_error(message: str, json_output: bool) -> None:
    if json_output:
        print(json_mod.dumps({'status': 'error', 'message': message}), file=sys.stderr)
    else:
        print(f'pxt: error: {message}', file=sys.stderr)


def _add_service_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--host', type=str, default=None, help='Bind address (overrides config default)')
    p.add_argument('--port', type=int, default=None, help='Bind port (overrides config default)')
    p.add_argument('--title', type=str, default=None, help='Service title (overrides config default)')
    p.add_argument('--prefix', type=str, default=None, help='URL prefix (overrides config default)')


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--dry-run', action='store_true', dest='dry_run', help='Print the resolved config without starting the server'
    )
    p.add_argument('--json', action='store_true', dest='json', help='Emit machine-readable JSON output')


def _add_serve_subparsers(serve_parser: argparse.ArgumentParser) -> None:
    serve_sub = serve_parser.add_subparsers(dest='mode', required=True)

    # pxt serve config <path>
    config_parser = serve_sub.add_parser(
        'config',
        help='Load service from a TOML config file',
        epilog=_EPILOG_CONFIG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_parser.add_argument('config', type=str, help='Path to the service TOML file')
    _add_service_args(config_parser)
    _add_output_args(config_parser)

    # pxt serve insert
    insert_parser = serve_sub.add_parser(
        'insert',
        help='Single insert endpoint',
        epilog=_EPILOG_INSERT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    insert_parser.add_argument('--table', required=True, help='Fully-qualified table path')
    insert_parser.add_argument('--path', required=True, help='HTTP path for the endpoint')
    insert_parser.add_argument('--inputs', nargs='+', default=None, help='Column names accepted from the JSON body')
    insert_parser.add_argument(
        '--uploadfile-inputs',
        nargs='+',
        default=None,
        dest='uploadfile_inputs',
        help='Column names accepted as multipart file uploads',
    )
    insert_parser.add_argument('--outputs', nargs='+', default=None, help='Column names returned in the response')
    insert_parser.add_argument(
        '--return-fileresponse',
        action='store_true',
        dest='return_fileresponse',
        help='Stream the output as a file response',
    )
    insert_parser.add_argument('--background', action='store_true', help='Run the insert in the background')
    _add_service_args(insert_parser)
    _add_output_args(insert_parser)

    # pxt serve delete
    delete_parser = serve_sub.add_parser(
        'delete',
        help='Single delete endpoint',
        epilog=_EPILOG_DELETE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    delete_parser.add_argument('--table', required=True, help='Fully-qualified table path')
    delete_parser.add_argument('--path', required=True, help='HTTP path for the endpoint')
    delete_parser.add_argument(
        '--match-columns',
        nargs='+',
        default=None,
        dest='match_columns',
        help='Column names used to match rows for deletion',
    )
    delete_parser.add_argument('--background', action='store_true', help='Run the delete in the background')
    _add_service_args(delete_parser)
    _add_output_args(delete_parser)

    # pxt serve query
    query_parser = serve_sub.add_parser(
        'query',
        help='Single query endpoint',
        epilog=_EPILOG_QUERY,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    query_parser.add_argument('--query', required=True, help='Dotted Python path to a @pxt.query or retrieval_udf')
    query_parser.add_argument('--path', required=True, help='HTTP path for the endpoint')
    query_parser.add_argument('--inputs', nargs='+', default=None, help='Parameter names accepted from the request')
    query_parser.add_argument(
        '--uploadfile-inputs',
        nargs='+',
        default=None,
        dest='uploadfile_inputs',
        help='Parameter names accepted as multipart file uploads',
    )
    query_parser.add_argument(
        '--one-row', action='store_true', dest='one_row', help='Return a single row instead of a list'
    )
    query_parser.add_argument(
        '--return-fileresponse',
        action='store_true',
        dest='return_fileresponse',
        help='Stream the output as a file response',
    )
    query_parser.add_argument('--background', action='store_true', help='Run the query in the background')
    query_parser.add_argument('--method', choices=['get', 'post'], default='post', help='HTTP method (default: post)')
    _add_service_args(query_parser)
    _add_output_args(query_parser)


def _serve(args: argparse.Namespace) -> None:
    from pixeltable.serving._config import AppConfig, ServiceConfig, create_app_from_config, load_app_config

    if args.mode == 'config':
        config = load_app_config(args.config)
    else:
        try:
            route = _build_route_from_args(args)
            config = AppConfig(service=ServiceConfig(), routes=[route])
        except pydantic.ValidationError as e:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, str(e)) from e

    overrides = {
        k: v
        for k, v in (('host', args.host), ('port', args.port), ('title', args.title), ('prefix', args.prefix))
        if v is not None
    }
    if overrides:
        try:
            new_service = ServiceConfig.model_validate(config.service.model_dump() | overrides)
        except pydantic.ValidationError as e:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, str(e)) from e
        config = config.model_copy(update={'service': new_service})

    if args.dry_run:
        _print_dry_run(config, args.json)
        return

    _run(config, create_app_from_config(config), args.json)


def _print_dry_run(config: 'AppConfig', json_output: bool) -> None:
    if json_output:
        print(config.model_dump_json(indent=2))
    else:
        svc = config.service
        print(f'Service:  {svc.title}')
        print(f'  Host:   {svc.host}')
        print(f'  Port:   {svc.port}')
        if svc.prefix:
            print(f'  Prefix: {svc.prefix}')
        print(f'Routes ({len(config.routes)}):')
        for route in config.routes:
            d = route.model_dump()
            print(f'  [{d["type"]}] {d["path"]}')


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


def _run(config: 'AppConfig', app: Any, json_output: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise excs.RequestError(
            excs.ErrorCode.MISSING_REQUIRED,
            "uvicorn is required for `pxt serve`; install it with `pip install 'fastapi[standard]'`",
        ) from e

    host, port = config.service.host, config.service.port
    # wildcard bind addresses aren't navigable; print localhost for the URL hints
    display_host = 'localhost' if host in ('0.0.0.0', '::', '') else host
    if ':' in display_host:
        display_host = f'[{display_host}]'
    url = f'http://{display_host}:{port}'
    docs_url = f'{url}/docs'

    if json_output:
        print(
            json_mod.dumps(
                {
                    'status': 'starting',
                    'host': host,
                    'port': port,
                    'url': url,
                    'docs_url': docs_url,
                    'routes': len(config.routes),
                }
            )
        )
    else:
        print(f'Starting Pixeltable service: {config.service.title}')
        print(f'  Bound to {host}:{port}')
        print(f'  Listening on {url}')
        print(f'  API docs at {docs_url}')
        print(f'  Routes: {len(config.routes)}')

    try:
        uvicorn.run(app, host=host, port=port)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            message = f'port {port} is already in use'
            if json_output:
                print(
                    json_mod.dumps({'status': 'error', 'code': 'EADDRINUSE', 'port': port, 'message': message}),
                    file=sys.stderr,
                )
            else:
                print(f'pxt: error: {message}', file=sys.stderr)
            sys.exit(1)
        raise
