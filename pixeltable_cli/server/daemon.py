"""`python -m pixeltable_cli.server.daemon` - pxt daemon entry point."""

import atexit
import os
import sys

from pixeltable_cli.client.utils import is_running
from pixeltable_cli.server.http_server import bind, run
from pixeltable_cli.utils import get_port, pidfile_path


def _write_pidfile() -> None:
    """Record our PID. The bound listen socket is the actual single-daemon mutex; the
    pidfile is only bookkeeping for tools like `pxt daemon status`."""
    path = pidfile_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(os.getpid()))


def _remove_pidfile_if_ours() -> None:
    path = pidfile_path()
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


def main() -> None:
    port = get_port()
    try:
        server = bind(port)
    except OSError as e:
        # Port held by something. If it's a peer pxt daemon, defer silently and let the
        # client's health probe find it. Otherwise this is a real failure — log it so the
        # client's `wait_for_health` log-tail surfaces something actionable.
        if is_running():
            sys.exit(0)
        print(f'pxt daemon: bind to 127.0.0.1:{port} failed: {e}', file=sys.stderr)
        sys.exit(1)
    _write_pidfile()
    atexit.register(_remove_pidfile_if_ours)
    run(server)


if __name__ == '__main__':
    main()
