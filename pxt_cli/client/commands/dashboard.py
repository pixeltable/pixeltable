from __future__ import annotations

import sys
import webbrowser

from pxt_cli import probe

from .. import http
from ..parser import Parser

EPILOG = """\
Examples:
  pxt dashboard start
  pxt dashboard start --no-open   # enable but don't launch a browser
  pxt dashboard stop
  pxt dashboard restart
  pxt dashboard open              # just print and open the URL
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt dashboard', description='control the dashboard SPA', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)
    start = sub.add_parser('start', help='enable the dashboard SPA; print URL and launch browser')
    start.add_argument('--no-open', action='store_true', dest='no_open', help='skip launching a browser')
    sub.add_parser('stop', help='disable the dashboard SPA (CLI traffic unaffected)')
    restart = sub.add_parser('restart', help='disable then enable')
    restart.add_argument('--no-open', action='store_true', dest='no_open', help='skip launching a browser')
    sub.add_parser('open', help='print and open the dashboard URL without changing the flag')

    args = parser.parse_args(argv)

    if args.action == 'start':
        _do_start(open_browser=not args.no_open)
    elif args.action == 'stop':
        _do_stop(ok_if_absent=False)
    elif args.action == 'restart':
        # Restart from cold (no daemon yet) means the flag was never set; treat the
        # missing daemon as already-disabled so the start path can spawn it.
        _do_stop(ok_if_absent=True)
        _do_start(open_browser=not args.no_open)
    elif args.action == 'open':
        _do_open()


def _do_start(open_browser: bool) -> None:
    http.post('/api/dashboard/control', {'action': 'enable'})
    url = probe.base_url()
    print(f'pxt dashboard enabled at {url}')
    if open_browser:
        webbrowser.open(url)


def _do_stop(ok_if_absent: bool) -> None:
    health = probe.fetch_health()
    if health is None:
        if ok_if_absent:
            return
        print('pxt: no daemon running', file=sys.stderr)
        sys.exit(1)
    http.post('/api/dashboard/control', {'action': 'disable'})
    print('pxt dashboard disabled')


def _do_open() -> None:
    # Match the start subcommand: spawn the daemon if it isn't already up so the browser
    # doesn't land on a connection error. The flag stays where it was; open is URL-launch,
    # not control.
    try:
        probe.ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)
    url = probe.base_url()
    print(url)
    webbrowser.open(url)
