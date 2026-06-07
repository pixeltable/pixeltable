from __future__ import annotations

import json
import os
import sys
from typing import Any

from pixeltable_cli.client.utils import ensure_running, fetch_health, kill_and_wait, read_pidfile
from pixeltable_cli.utils import get_port, pidfile_path

from ..parser import Parser

EPILOG = """\
Examples:
  pxt daemon start
  pxt daemon stop
  pxt daemon stop --force         # bypass the responder-PID safety check
  pxt daemon restart
  pxt daemon status
  pxt daemon status --json
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt daemon', description='control the pxt daemon', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('start', help='ensure the daemon is running; print URL and PID')
    stop = sub.add_parser('stop', help='terminate the daemon we own')
    stop.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='kill whichever PID is responding on the port even if it does not match the pidfile',
    )
    sub.add_parser('restart', help='stop then start')
    status = sub.add_parser('status', help='print daemon info; exit 1 if no daemon is running')
    status.add_argument('--json', action='store_true', dest='as_json', help='emit the identity fingerprint as JSON')

    args = parser.parse_args(argv)

    if args.action == 'start':
        _do_start()
    elif args.action == 'stop':
        _do_stop(force=args.force)
    elif args.action == 'restart':
        _do_stop(force=False, ok_if_absent=True)
        _do_start()
    elif args.action == 'status':
        _do_status(as_json=args.as_json)


def _do_start() -> None:
    try:
        base = ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)
    health = fetch_health()
    pid = health.get('pid') if health is not None else None
    suffix = f' (PID {pid})' if pid is not None else ''
    print(f'pxt daemon up at {base}{suffix}')


def _do_stop(force: bool, ok_if_absent: bool = False) -> None:
    tracked_pid = read_pidfile()
    health = fetch_health()

    if tracked_pid is None and health is None:
        if ok_if_absent:
            return
        print('pxt: no daemon running', file=sys.stderr)
        sys.exit(1)

    # Pick the PID to kill: by default the pidfile's; with --force, fall back to whatever is
    # responding on the port. Refuse the mismatch case unless the user opted in.
    pid_to_kill: int | None
    if health is not None and tracked_pid is not None and health.get('pid') == tracked_pid:
        pid_to_kill = tracked_pid
    elif health is not None and tracked_pid is not None and health.get('pid') != tracked_pid:
        if not force:
            print(
                f'pxt: responder on port {get_port()} (PID {health.get("pid")}) does not match '
                f'pidfile (PID {tracked_pid}); refusing to kill it. Use --force to override.',
                file=sys.stderr,
            )
            sys.exit(1)
        pid_to_kill = health.get('pid')
    elif health is None and tracked_pid is not None:
        # No responder but we have a pidfile entry. Could be a hung daemon (PID alive but not
        # listening) or a stale pidfile (PID gone).
        pid_to_kill = tracked_pid
    else:
        # health is not None, tracked_pid is None: no pidfile but something is responding.
        # Refuse unless forced - we have no proof the responder is ours.
        if not force:
            print(
                f'pxt: a process on port {get_port()} (PID {health.get("pid")}) is responding but '
                f'no pidfile claims it. Refusing to kill it. Use --force to override.',
                file=sys.stderr,
            )
            sys.exit(1)
        pid_to_kill = health.get('pid')

    if pid_to_kill is not None:
        kill_and_wait(pid_to_kill)
    # Daemon's atexit handler only fires on graceful exit; clean up the pidfile here so a
    # subsequent stop doesn't trip the "stale pidfile" branch.
    try:
        os.remove(pidfile_path())
    except OSError:
        pass
    print(f'pxt: stopped daemon (PID {pid_to_kill})')


def _do_status(as_json: bool) -> None:
    health = fetch_health()
    if health is None:
        print('pxt: no daemon running', file=sys.stderr)
        sys.exit(1)

    if as_json:
        print(json.dumps(health, indent=2))
        return

    _print_status_text(health)


def _print_status_text(health: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = [
        ('PID', str(health.get('pid'))),
        ('Started', str(health.get('started_at'))),
        ('Service', str(health.get('service'))),
        ('Version', str(health.get('pxt_version'))),
        ('Install dir', str(health.get('pxt_install_dir'))),
        ('Python', str(health.get('python_executable'))),
        ('Home', str(health.get('pixeltable_home'))),
        ('PgData', str(health.get('pixeltable_pgdata'))),
        ('Config', str(health.get('pixeltable_config_file'))),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f'{k.ljust(width)}  {v}')

    env_snapshot = health.get('pixeltable_env') or {}
    if len(env_snapshot) > 0:
        print()
        print('PIXELTABLE_* env vars at daemon startup:')
        for k in sorted(env_snapshot):
            print(f'  {k}={env_snapshot[k]}')
