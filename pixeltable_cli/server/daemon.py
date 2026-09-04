"""`python -m pixeltable_cli.server.daemon` - pxt daemon entry point."""

import argparse
import atexit
import os
import pathlib
import sys

from pixeltable.config import Config
from pixeltable_cli.client.utils import is_running
from pixeltable_cli.server.http_server import bind, run
from pixeltable_cli.utils import get_port, pidfile_path

_LOOPBACK = '127.0.0.1'


def _write_pidfile(port: int) -> None:
    """Record our PID. The bound listen socket is the actual single-daemon mutex; the
    pidfile is only bookkeeping for tools like `pxt daemon status`."""
    path = pidfile_path(port)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(os.getpid()))


def _remove_pidfile_if_ours(port: int) -> None:
    path = pidfile_path(port)
    try:
        with open(path, encoding='utf-8') as f:
            owner = int(f.read().strip())
    except (OSError, ValueError):
        return
    if owner != os.getpid():
        return
    try:
        os.remove(path)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog='pxt daemon', description=__doc__)
    ap.add_argument('--project-root', type=pathlib.Path, default=None)
    ap.add_argument('--host', default=_LOOPBACK, help='listen address; the API it serves is unauthenticated')
    ap.add_argument('--port', type=int, default=None, help='listen port; defaults to PXT_PORT')
    args = ap.parse_args(argv)

    named_address = args.port is not None or args.host != _LOOPBACK
    port = get_port() if args.port is None else args.port
    if named_address and args.host != _LOOPBACK:
        # every route is unauthenticated, including the destructive ones; only a proxy that authenticates
        # its callers may sit in front of a daemon reachable beyond this machine
        print(f'pxt daemon: serving the unauthenticated API on {args.host}:{port}', file=sys.stderr)
    try:
        server = bind(args.host, port)
    except OSError as e:
        # Port held by something. If it's a peer pxt daemon on the port this machine's clients use, defer
        # silently and let the client's health probe find it; a caller that named an address gets the
        # failure, since nothing tells it that another daemon answers elsewhere.
        if not named_address and is_running():
            sys.exit(0)
        print(f'pxt daemon: bind to {args.host}:{port} failed: {e}', file=sys.stderr)
        sys.exit(1)
    _write_pidfile(port)
    atexit.register(lambda: _remove_pidfile_if_ours(port))
    Config.init(reinit=True, project_root=args.project_root)
    run(server)


if __name__ == '__main__':
    main()
