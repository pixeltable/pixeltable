"""Pixeltable CLI entry point."""

from __future__ import annotations

import argparse
import errno
import json
import sys
from typing import Any

import pydantic

import pixeltable as pxt
from pixeltable import config, exceptions as excs
from pixeltable.serving._config import create_service_from_config, lookup_service_config


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that appends the epilog examples to stderr on error."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        sys.stderr.write(f'\npxt: error: {message}\n')
        if self.epilog is not None:
            sys.stderr.write(f'\n{self.epilog}\n')
        sys.exit(2)


_SERVE_SUBCOMMANDS = ('insert', 'query', 'update', 'delete')

_EPILOG_SERVE = """\
To start a configured service:
  pxt serve <service-name>
  pxt serve <service-name> --port 9000"""

_EPILOG_INSERT = """\
Examples:
  pxt serve insert --table my_dir.my_table --path /generate \\
    --inputs prompt --outputs prompt result --port 8000"""

_EPILOG_UPDATE = """\
Examples:
  pxt serve update --table my_dir.my_table --path /update \\
    --inputs prompt --outputs id result --port 8000"""

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

    serve_parser = subparsers.add_parser(
        'serve', help='Start an HTTP service', formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Detect whether this is `pxt serve <service-name>` or `pxt serve <subcommand>`.
    # If the first arg after `serve` is not a known subcommand and not a flag,
    # set up the parser for named-service mode; otherwise set up subcommand parsers.
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == 'serve' and argv[1] not in _SERVE_SUBCOMMANDS and not argv[1].startswith('-'):
        serve_parser.add_argument('service', help='Name of the configured service to start')
        serve_parser.add_argument('--config', type=str, default=None, help='Path to an additional TOML config file')
        _add_service_args(serve_parser)
        _add_output_args(serve_parser)
    else:
        serve_parser.epilog = _EPILOG_SERVE
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
        print(json.dumps({'status': 'error', 'message': message}), file=sys.stderr)
    else:
        print(f'pxt: error: {message}', file=sys.stderr)


def _add_service_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--host', type=str, default=None, help='Bind address (overrides config default)')
    p.add_argument('--port', type=int, default=None, help='Bind port (overrides config default)')
    p.add_argument('--prefix', type=str, default=None, help='URL prefix (overrides config default)')


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--dry-run', action='store_true', dest='dry_run', help='Print the resolved config without starting the server'
    )
    p.add_argument('--json', action='store_true', dest='json', help='Emit machine-readable JSON output')


def _add_serve_subparsers(serve_parser: argparse.ArgumentParser) -> None:
    serve_sub = serve_parser.add_subparsers(dest='mode', required=True)

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

    # pxt serve update
    update_parser = serve_sub.add_parser(
        'update',
        help='Single update endpoint',
        epilog=_EPILOG_UPDATE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    update_parser.add_argument('--table', required=True, help='Fully-qualified table path')
    update_parser.add_argument('--path', required=True, help='HTTP path for the endpoint')
    update_parser.add_argument(
        '--inputs',
        nargs='+',
        default=None,
        help='Non-PK column names accepted from the JSON body (PK columns are always accepted)',
    )
    update_parser.add_argument('--outputs', nargs='+', default=None, help='Column names returned in the response')
    update_parser.add_argument(
        '--return-fileresponse',
        action='store_true',
        dest='return_fileresponse',
        help='Stream the output as a file response',
    )
    update_parser.add_argument('--background', action='store_true', help='Run the update in the background')
    _add_service_args(update_parser)
    _add_output_args(update_parser)

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
    if hasattr(args, 'service'):
        if args.config is not None:
            config.Config.init({}, additional_config_files=[args.config])
        cfg = lookup_service_config(args.service)
    else:
        try:
            route = _build_route_from_args(args)
            cfg = config.ServiceConfig(name='pxt-serve', routes=[route])
        except pydantic.ValidationError as e:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, str(e)) from e

    overrides = {k: v for k, v in (('host', args.host), ('port', args.port), ('prefix', args.prefix)) if v is not None}
    if overrides:
        try:
            cfg = config.ServiceConfig.model_validate(cfg.model_dump() | overrides)
        except pydantic.ValidationError as e:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, str(e)) from e

    if args.dry_run:
        _print_dry_run(cfg, args.json)
        return

    _run(cfg, create_service_from_config(cfg), args.json)


def _print_dry_run(config: config.ServiceConfig, json_output: bool) -> None:
    if json_output:
        print(config.model_dump_json(indent=2))
    else:
        print(f'Service:  {config.name}')
        print(f'  Host:   {config.host}')
        print(f'  Port:   {config.port}')
        if config.prefix:
            print(f'  Prefix: {config.prefix}')
        print(f'Routes ({len(config.routes)}):')
        for route in config.routes:
            d = route.model_dump()
            print(f'  [{d["type"]}] {d["path"]}')


def _build_route_from_args(args: argparse.Namespace) -> config.RouteConfig:
    if args.mode == 'insert':
        return config.InsertRouteConfig(
            type='insert',
            table=args.table,
            path=args.path,
            inputs=args.inputs,
            uploadfile_inputs=args.uploadfile_inputs,
            outputs=args.outputs,
            return_fileresponse=args.return_fileresponse,
            background=args.background,
        )
    if args.mode == 'update':
        return config.UpdateRouteConfig(
            type='update',
            table=args.table,
            path=args.path,
            inputs=args.inputs,
            outputs=args.outputs,
            return_fileresponse=args.return_fileresponse,
            background=args.background,
        )
    if args.mode == 'delete':
        return config.DeleteRouteConfig(
            type='delete',
            table=args.table,
            path=args.path,
            match_columns=args.match_columns,
            background=args.background,
        )
    if args.mode == 'query':
        return config.QueryRouteConfig(
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


def _run(config: config.ServiceConfig, app: Any, json_output: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise excs.RequestError(
            excs.ErrorCode.MISSING_REQUIRED,
            "uvicorn is required for `pxt serve`; install it with `pip install 'fastapi[standard]'`",
        ) from e

    host, port = config.host, config.port
    # wildcard bind addresses aren't navigable; print localhost for the URL hints
    display_host = 'localhost' if host in ('0.0.0.0', '::', '') else host
    if ':' in display_host:
        display_host = f'[{display_host}]'
    url = f'http://{display_host}:{port}'
    docs_url = f'{url}/docs'

    if json_output:
        print(
            json.dumps(
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
        print(f'Starting Pixeltable service: {config.name}')
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
                    json.dumps({'status': 'error', 'code': 'EADDRINUSE', 'port': port, 'message': message}),
                    file=sys.stderr,
                )
            else:
                print(f'pxt: error: {message}', file=sys.stderr)
            sys.exit(1)
        raise
