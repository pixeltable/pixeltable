"""`pxt localproxy {create,start,stop,delete} <db>` — manage local-proxy daemons."""

from __future__ import annotations

from typing import NoReturn

from ..parser import Parser
from ..utils import fail

EPILOG = """\
Examples:
  pxt localproxy create testdb     # create the daemon's home (db is created on first start)
  pxt localproxy start testdb      # start the daemon; print its endpoint
  pxt localproxy stop testdb       # stop the daemon
  pxt localproxy delete testdb     # stop, drop the database, remove the home

Notes:
  stop and delete exit 1 if there is no local proxy with that name.
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt localproxy', description='manage local proxy daemons', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('create', 'start', 'stop', 'delete'):
        p = sub.add_parser(action, help=f'{action} a local proxy daemon')
        p.add_argument('db', help='the proxy database name (the <db> in pxt://local:<db>/...)')
    args = parser.parse_args(argv)

    from pixeltable.service import proxy_daemon

    def fail_not_found() -> NoReturn:
        fail(
            f'pxt localproxy {args.action}',
            f'no local proxy named {args.db!r} (looked in {proxy_daemon.proxy_home(args.db)})',
            'DEPLOYMENT_NOT_FOUND',
            as_json=False,
        )

    if args.action == 'create':
        proxy_daemon.create(args.db)
        print(f'Created local proxy {args.db!r}.')
    elif args.action == 'start':
        endpoint = proxy_daemon.start(args.db)
        print(f'Local proxy {args.db!r} running at {endpoint}')
    elif args.action == 'stop':
        if not proxy_daemon.proxy_home(args.db).exists():
            fail_not_found()
        running = proxy_daemon.endpoint(args.db) is not None
        proxy_daemon.stop(args.db)
        print(f'Stopped local proxy {args.db!r}.' if running else f'Local proxy {args.db!r} is not running.')
    elif args.action == 'delete':
        if not proxy_daemon.delete(args.db):
            fail_not_found()
        print(f'Deleted local proxy {args.db!r}.')
