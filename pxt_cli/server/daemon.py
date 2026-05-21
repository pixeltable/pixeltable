"""`python -m pxt_cli.server.daemon` - pxt daemon entry point."""

import atexit
import os
import sys

from pxt_cli.server.http_server import bind, run
from pxt_cli.utils import get_port, pidfile_path


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
    try:
        server = bind(get_port())
    except OSError:
        # Port already taken by another daemon (or process). Defer.
        sys.exit(0)
    _write_pidfile()
    atexit.register(_remove_pidfile_if_ours)
    run(server)


if __name__ == '__main__':
    main()
