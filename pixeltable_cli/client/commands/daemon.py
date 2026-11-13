from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any

import psutil

from pixeltable_cli.client.utils import (
    describe_in_flight_requests,
    ensure_running,
    fetch_health,
    kill_and_wait,
    pid_is_our_daemon,
    plural,
    port_is_open,
    read_pidfile,
)
from pixeltable_cli.utils import get_port, pidfile_path, working_dirs_path

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
        help='kill the daemon even if it is serving requests, or if the PID responding on the port does not '
        'match the pidfile',
    )
    restart = sub.add_parser('restart', help='stop then start')
    restart.add_argument('-f', '--force', action='store_true', help='restart even while the daemon is serving requests')
    status = sub.add_parser('status', help='print daemon info; exit 1 if no daemon is running')
    status.add_argument('--json', action='store_true', dest='as_json', help='emit the identity fingerprint as JSON')

    args = parser.parse_args(argv)

    if args.action == 'start':
        _start()
    elif args.action == 'stop':
        _stop(force=args.force)
        # a stop clears the working directories set with pxt cd; a restart keeps them
        with contextlib.suppress(OSError):
            os.remove(working_dirs_path(get_port()))
    elif args.action == 'restart':
        _stop(force=args.force, ok_if_absent=True)
        _start()
    elif args.action == 'status':
        _status(as_json=args.as_json)


def _start() -> None:
    try:
        base = ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)
    health = fetch_health()
    pid = health.get('pid') if health is not None else None
    suffix = f' (PID {pid})' if pid is not None else ''
    print(f'pxt daemon up at {base}{suffix}')


def _stop(force: bool, ok_if_absent: bool = False) -> None:
    tracked_pid = read_pidfile()
    health = fetch_health()

    in_flight = health.get('in_flight', []) if health is not None else []
    if len(in_flight) > 0 and not force:
        print(
            f'pxt: the daemon is serving {plural(len(in_flight), "request")}: '
            f'{describe_in_flight_requests(in_flight)}.\n'
            'Wait for it to finish, or use --force.',
            file=sys.stderr,
        )
        sys.exit(1)

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
        # listening) or a stale pidfile, whose PID is gone or now belongs to an unrelated process.
        pid_to_kill = tracked_pid if pid_is_our_daemon(tracked_pid) else None
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
        with contextlib.suppress(psutil.NoSuchProcess, OSError):  # a daemon already gone needs no kill
            kill_and_wait(psutil.Process(pid_to_kill))
    # Daemon's atexit handler only fires on graceful exit; clean up the pidfile here so a
    # subsequent stop doesn't trip the "stale pidfile" branch.
    try:
        os.remove(pidfile_path())
    except OSError:
        pass
    if pid_to_kill is None:
        print(f'pxt: no daemon running; removed a stale pidfile (PID {tracked_pid})')
    else:
        print(f'pxt: stopped daemon (PID {pid_to_kill})')


def _status(as_json: bool) -> None:
    # a longer timeout than the default: a daemon serving a slow request answers late, and reporting it as
    # absent would be wrong
    health = fetch_health(timeout=5.0)
    if health is None:
        if port_is_open():
            print(
                'pxt: the daemon is not answering; it is busy with a request. '
                "Run 'pxt daemon status' again, or 'pxt daemon stop -f' to take it down.",
                file=sys.stderr,
            )
        else:
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
        ('Project', str(health.get('project_root'))),
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
