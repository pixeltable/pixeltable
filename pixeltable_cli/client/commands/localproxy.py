"""`pxt localproxy {create,start,stop,delete} <db>` — manage local-proxy daemons."""

from __future__ import annotations

from ..parser import Parser
from ..utils import fail

EPILOG = """\
Examples:
  pxt localproxy create testdb     # create the daemon's home (db is created on first start)
  pxt localproxy start testdb      # start the daemon; print its endpoint
  pxt localproxy stop testdb       # stop the daemon
  pxt localproxy delete testdb     # stop, drop the database, remove the home

Notes:
  stop and delete exit 1 for a name with no local proxy, so a typo does not look like success.
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt localproxy', description='manage local proxy daemons', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('create', 'start', 'stop', 'delete'):
        p = sub.add_parser(action, help=f'{action} a local proxy daemon')
        p.add_argument('db', help='the proxy database name (the <db> in pxt://local:<db>/...)')
    args = parser.parse_args(argv)

    from pixeltable.service import proxy_daemon

    prog = f'pxt localproxy {args.action}'
    not_found = f'no local proxy named {args.db!r} (looked in {proxy_daemon.proxy_home(args.db)})'
    if args.action == 'create':
        proxy_daemon.create(args.db)
        print(f'Created local proxy {args.db!r}.')
    elif args.action == 'start':
        endpoint = proxy_daemon.start(args.db)
        print(f'Local proxy {args.db!r} running at {endpoint}')
    elif args.action == 'stop':
        if not proxy_daemon.proxy_home(args.db).exists():
            fail(prog, not_found, 'DEPLOYMENT_NOT_FOUND', as_json=False)
        running = proxy_daemon.endpoint(args.db) is not None
        proxy_daemon.stop(args.db)
        print(f'Stopped local proxy {args.db!r}.' if running else f'Local proxy {args.db!r} is not running.')
    elif args.action == 'delete':
        if not proxy_daemon.delete(args.db):
            fail(prog, not_found, 'DEPLOYMENT_NOT_FOUND', as_json=False)
        print(f'Deleted local proxy {args.db!r}.')
