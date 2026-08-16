"""`pxt service {create,update,list,status,start,stop,delete} <uri>` - manage hosted services."""

from __future__ import annotations

import argparse
import json

from ..hosted import exit_unless_reached, parse_base_uri, parse_db_uri, parse_service_uri, poll_svc, print_service
from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt service create NAME --base-uri pxt://org:db
  pxt service update pxt://org:db/services/NAME --workers 2
  pxt service list pxt://org:db
  pxt service status pxt://org:db/services/NAME
  pxt service start pxt://org:db/services/NAME
  pxt service stop pxt://org:db/services/NAME
  pxt service delete pxt://org:db/services/NAME
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service', description='manage hosted services', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('create', help='create a service from a table in a hosted database')
    p.add_argument('name', help='Service name (must match a [[pixeltable.service]] block in the config)')
    p.add_argument(
        '--base-uri',
        required=True,
        dest='base_uri',
        metavar='URI',
        help='pxt://org:db[/<dir>] - database and base path prefix for resolving relative table paths in routes',
    )
    p.add_argument('--workers', type=int, default=1, help='Number of workers (default: 1)')
    p.add_argument('--cpu', type=float, default=0.5, help='CPU cores per worker (default: 0.5)')
    p.add_argument('--memory', type=int, default=512, dest='memory_mb', help='Memory per worker in MB (default: 512)')
    p.add_argument('--disk', type=int, default=10, dest='disk_gb', help='Disk per worker in GB (default: 10)')
    p.add_argument('--config', default=None, metavar='FILE', help='Path to an additional config file (TOML)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('update', help='update service routes or worker count')
    p.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    p.add_argument('--workers', type=int, default=None, help='New minimum worker count')
    p.add_argument('--cpu', type=float, default=None, help='CPU cores per worker')
    p.add_argument('--memory', type=int, default=None, dest='memory_mb', help='Memory per worker in MB')
    p.add_argument('--disk', type=int, default=None, dest='disk_gb', help='Disk per worker in GB')
    p.add_argument('--config', default=None, metavar='FILE', help='Path to an additional config file (TOML)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('list', help='list services in a hosted database')
    p.add_argument('db_uri', help='Database URI: pxt://org:db')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of a service')
    p.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('start', help='start a service')
    p.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('stop', help='stop a service')
    p.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('delete', help='delete a service')
    p.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'create':
        _create(args)
    elif args.action == 'update':
        _update(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)
    elif args.action == 'start':
        _start(args)
    elif args.action == 'stop':
        _stop(args)
    elif args.action == 'delete':
        _delete(args)


def _create(args: argparse.Namespace) -> None:
    # imported lazily to keep the other service subcommands free of the pixeltable import
    from pixeltable import config as pxt_config
    from pixeltable.serving._config import lookup_service_config

    org, db, base_path = parse_base_uri(args.base_uri, prog='pxt service create')

    if args.config is not None:
        pxt_config.Config.init({}, additional_config_files=[args.config])
    service_config = lookup_service_config(args.name).model_dump_json()

    resp = post_request(
        '/api/services',
        {
            'org': org,
            'db': db,
            'service_name': args.name,
            'base_path': base_path,
            'workers_min': args.workers,
            'cpu': args.cpu,
            'memory_mb': args.memory_mb,
            'disk_gb': args.disk_gb,
            'service_config': service_config,
        },
    )
    svc = resp.get('service', resp) if isinstance(resp, dict) else {}
    if svc.get('state') in ('DEPLOYING', 'STARTING'):
        svc = poll_svc(org, db, args.name, {'DEPLOYING', 'STARTING'}, f"Service '{args.name}' is deploying...")
    if args.json_output:
        print(json.dumps(svc))
    else:
        print_service(svc)
    exit_unless_reached(svc, 'AVAILABLE', f'deploying service {args.name!r}')


def _update(args: argparse.Namespace) -> None:
    # imported lazily to keep the other service subcommands free of the pixeltable import
    from pixeltable import config as pxt_config, exceptions as excs
    from pixeltable.serving._config import lookup_service_config

    org, db, svc_name = parse_service_uri(args.service_uri, prog='pxt service update')

    additional_files = [args.config] if args.config is not None else []
    pxt_config.Config.init({}, additional_config_files=additional_files)
    try:
        service_config = lookup_service_config(svc_name).model_dump_json()
    except excs.NotFoundError:
        service_config = None

    resp = post_request(
        '/api/service/update',
        {
            'org': org,
            'db': db,
            'service_name': svc_name,
            'workers_min': args.workers,
            'cpu': args.cpu,
            'memory_mb': args.memory_mb,
            'disk_gb': args.disk_gb,
            'service_config': service_config,
        },
    )
    svc = resp.get('service', resp) if isinstance(resp, dict) else {}
    if svc.get('state') == 'UPDATING':
        svc = poll_svc(org, db, svc_name, {'UPDATING'}, f"Service '{svc_name}' is updating...")
    if args.json_output:
        print(json.dumps(svc))
    else:
        print_service(svc)
    exit_unless_reached(svc, 'AVAILABLE', f'updating service {svc_name!r}')


def _list(args: argparse.Namespace) -> None:
    org, db = parse_db_uri(args.db_uri, prog='pxt service list')
    resp = get_request('/api/services', {'org': org, 'db': db})
    svcs = resp.get('services', []) if isinstance(resp, dict) else []
    if args.json_output:
        print(json.dumps(svcs))
    elif not svcs:
        print(f"No services in database '{db}'.")
    else:
        for svc in svcs:
            print_service(svc)


def _status(args: argparse.Namespace) -> None:
    org, db, svc_name = parse_service_uri(args.service_uri, prog='pxt service status')
    resp = get_request('/api/service', {'org': org, 'db': db, 'service_name': svc_name})
    svc = resp.get('service', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(svc))
    else:
        print_service(svc)


def _start(args: argparse.Namespace) -> None:
    org, db, svc_name = parse_service_uri(args.service_uri, prog='pxt service start')
    resp = post_request('/api/service/start', {'org': org, 'db': db, 'service_name': svc_name})
    svc = resp.get('service', resp) if isinstance(resp, dict) else {}
    if svc.get('state') in ('STARTING', 'DEPLOYING'):
        svc = poll_svc(org, db, svc_name, {'STARTING', 'DEPLOYING'}, f"Service '{svc_name}' is starting...")
    if args.json_output:
        print(json.dumps(svc))
    else:
        print_service(svc)
    exit_unless_reached(svc, 'AVAILABLE', f'starting service {svc_name!r}')


def _stop(args: argparse.Namespace) -> None:
    org, db, svc_name = parse_service_uri(args.service_uri, prog='pxt service stop')
    resp = post_request('/api/service/stop', {'org': org, 'db': db, 'service_name': svc_name})
    svc = resp.get('service', resp) if isinstance(resp, dict) else {}
    if svc.get('state') == 'STOPPING':
        svc = poll_svc(org, db, svc_name, {'STOPPING'}, f"Service '{svc_name}' is stopping...")
    if args.json_output:
        print(json.dumps(svc))
    else:
        print_service(svc)
    exit_unless_reached(svc, 'STOPPED', f'stopping service {svc_name!r}')


def _delete(args: argparse.Namespace) -> None:
    org, db, svc_name = parse_service_uri(args.service_uri, prog='pxt service delete')
    post_request('/api/service/delete', {'org': org, 'db': db, 'service_name': svc_name})
    if args.json_output:
        print(json.dumps({'deleted': svc_name}))
    else:
        print(f"Deleted service '{svc_name}'.")
