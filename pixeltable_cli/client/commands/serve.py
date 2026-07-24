from __future__ import annotations

import argparse
import errno
import json
import socket
import sys
from typing import Any

import pydantic

import pixeltable as pxt
from pixeltable import config, exceptions as excs
from pixeltable.env import Env
from pixeltable.serving._config import create_service_from_config, lookup_service_config

from ..parser import Parser

_SUBCOMMANDS = ('insert', 'query', 'update', 'delete')

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


def _add_service_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--host', type=str, default=None, help='Bind address (overrides config default)')
    p.add_argument('--port', type=int, default=None, help='Bind port (overrides config default)')
    p.add_argument('--prefix', type=str, default=None, help='URL prefix (overrides config default)')
    p.add_argument('--config', type=str, default=None, help='Path to an additional TOML config file')
    p.add_argument(
        '--otel',
        action='store_true',
        help="Enable OpenTelemetry instrumentation (requires `pip install 'pixeltable[otel]'`)",
    )


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--dry-run', action='store_true', dest='dry_run', help='Print the resolved config without starting the server'
    )
    p.add_argument('--json', action='store_true', dest='json', help='Emit machine-readable JSON output')


def _add_export_sql_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--export-sql-db-connect',
        dest='export_sql_db_connect',
        default=None,
        help='SQLAlchemy connection string for an external SQL target (enables export_sql)',
    )
    p.add_argument(
        '--export-sql-table',
        dest='export_sql_table',
        default=None,
        help='Target table name (required when --export-sql-db-connect is set)',
    )
    p.add_argument(
        '--export-sql-db-schema',
        dest='export_sql_db_schema',
        default=None,
        help='Optional database schema qualifier for the target table',
    )
    p.add_argument(
        '--export-sql-method',
        dest='export_sql_method',
        choices=('insert', 'update', 'merge'),
        default='insert',
        help="How to write each row into the target table (default: 'insert')",
    )


def _add_subparsers(serve_parser: argparse.ArgumentParser) -> None:
    serve_sub = serve_parser.add_subparsers(dest='mode', required=True)

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
    _add_export_sql_args(insert_parser)
    _add_service_args(insert_parser)
    _add_output_args(insert_parser)

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
    _add_export_sql_args(update_parser)
    _add_service_args(update_parser)
    _add_output_args(update_parser)

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


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt serve', description='Start an HTTP service')

    # Decide between named-service mode (pxt serve <service-name>) and subcommand mode
    # (pxt serve insert|update|delete|query) by inspecting the first argument: if it's
    # not a known subcommand and not a flag, treat it as a service name.
    if len(argv) >= 1 and argv[0] not in _SUBCOMMANDS and not argv[0].startswith('-'):
        parser.add_argument('service', help='Name of the configured service to start')
        parser.add_argument(
            '--base-uri',
            default=None,
            dest='base_uri',
            metavar='PATH',
            help='Base path prefix for resolving relative table paths in route config',
        )
        _add_service_args(parser)
        _add_output_args(parser)
    else:
        parser.epilog = _EPILOG_SERVE
        _add_subparsers(parser)

    args = parser.parse_args(argv)

    try:
        _serve(args)
    except pxt.Error as e:
        _emit_error(str(e), args.json)
        sys.exit(1)


def _serve(args: argparse.Namespace) -> None:
    if args.config is not None:
        config.Config.init({}, additional_config_files=[args.config])
    if hasattr(args, 'service'):
        cfg = lookup_service_config(args.service)
    else:
        try:
            route = _create_route_from_args(args)
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

    base_path = getattr(args, 'base_uri', None) or ''
    app = create_service_from_config(cfg, base_path=base_path)
    if args.otel:
        Env.get().require_package(
            'opentelemetry.instrumentation.pixeltable',
            not_installed_msg="--otel requires the instrumentation package; install: `pip install 'pixeltable[otel]'`",
        )
        import opentelemetry.instrumentation.pixeltable as pxt_otel

        pxt_otel.init()
        pxt_otel.instrument_fastapi(app)

    _run(cfg, app, args.json)


def _print_dry_run(cfg: config.ServiceConfig, json_output: bool) -> None:
    if json_output:
        print(cfg.model_dump_json(indent=2))
        return
    print(f'Service:  {cfg.name}')
    print(f'  Host:   {cfg.host}')
    print(f'  Port:   {cfg.port}')
    if cfg.prefix:
        print(f'  Prefix: {cfg.prefix}')
    print(f'Routes ({len(cfg.routes)}):')
    for route in cfg.routes:
        d = route.model_dump()
        print(f'  [{d["type"]}] {d["path"]}')


def _create_sql_export(args: argparse.Namespace) -> config.SqlExport | None:
    db_connect = args.export_sql_db_connect
    table = args.export_sql_table
    db_schema = args.export_sql_db_schema
    if db_connect is None:
        if table is not None or db_schema is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                '--export-sql-table / --export-sql-db-schema requires --export-sql-db-connect',
            )
        return None
    if table is None:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '--export-sql-db-connect requires --export-sql-table')
    return config.SqlExport(db_connect=db_connect, table=table, db_schema=db_schema, method=args.export_sql_method)


def _create_route_from_args(args: argparse.Namespace) -> config.RouteConfig:
    if args.mode == 'insert':
        return config.InsertRouteConfig(
            type='insert',
            table=args.table,
            path=args.path,
            inputs=args.inputs,
            uploadfile_inputs=args.uploadfile_inputs,
            outputs=args.outputs,
            return_fileresponse=args.return_fileresponse,
            export_sql=_create_sql_export(args),
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
            export_sql=_create_sql_export(args),
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


def _started_status(host: str, port: int, is_ssl: bool, n_routes: int, json_output: bool) -> str:
    # wildcard bind addresses aren't navigable; show localhost for the URL hints
    display_host = 'localhost' if host in ('0.0.0.0', '::') else host
    if ':' in display_host:
        display_host = f'[{display_host}]'

    scheme = 'https' if is_ssl else 'http'
    url = f'{scheme}://{display_host}:{port}'
    docs_url = f'{url}/docs'
    if json_output:
        return json.dumps(
            {'status': 'started', 'host': host, 'port': port, 'url': url, 'docs_url': docs_url, 'routes': n_routes}
        )
    return f'Pixeltable is running on {url}\n  Routes: {n_routes}\n  API docs at {docs_url}'


def _run(cfg: config.ServiceConfig, app: Any, json_output: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise excs.RequestError(
            excs.ErrorCode.MISSING_REQUIRED,
            "uvicorn is required for `pxt serve`; install it with `pip install 'pixeltable[serve]'`",
        ) from e

    class PxtServer(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            # Validate the server's configuration -- we would rather fail than print an unexpected or incorrect status.
            assert sockets is None
            await super().startup(sockets)
            assert self.config.fd is None
            assert self.config.uds is None
            if self.started:
                assert len(self.servers) == 1
                server = self.servers[0]
                assert len(server.sockets) == 1
                port = server.sockets[0].getsockname()[1]
                is_ssl = self.config.ssl is not None
                print(_started_status(self.config.host, port, is_ssl, len(cfg.routes), json_output))

    if not json_output:
        print(f'Starting Pixeltable service {cfg.name}...')
    # log_config=None keeps uvicorn from installing its own stderr handlers, we don't want its logging in the console.
    # Env routes uvicorn loggers to a file.
    server = PxtServer(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_config=None))
    try:
        server.run()
    except KeyboardInterrupt:
        print('Keyboard interrupt received, shutting down', file=sys.stderr)
        sys.exit(130)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            message = f'port {cfg.port} is already in use'
            if json_output:
                print(
                    json.dumps({'status': 'error', 'code': 'EADDRINUSE', 'port': cfg.port, 'message': message}),
                    file=sys.stderr,
                )
            else:
                print(f'pxt: error: {message}', file=sys.stderr)
            sys.exit(1)
        raise


def _emit_error(message: str, json_output: bool) -> None:
    if json_output:
        print(json.dumps({'status': 'error', 'message': message}), file=sys.stderr)
    else:
        print(f'pxt: error: {message}', file=sys.stderr)
