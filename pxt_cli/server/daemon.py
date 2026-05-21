"""`python -m pxt_cli.server.daemon` - pxt daemon entry point."""

import atexit
import os
import socket
import sys

from pxt_cli.server.http_server import serve
from pxt_cli.utils import get_port, pidfile_path


def _port_in_use() -> bool:
    """Whether anything is listening on our daemon port. Used in place of a PID-liveness
    check so zombies (still 'alive' to os.kill until reaped) don't masquerade as live peers."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.1)
    try:
        s.connect(('127.0.0.1', get_port()))
    except OSError:
        return False
    finally:
        s.close()
    return True


def _claim_pidfile() -> bool:
    """Atomically claim the pidfile. Returns False if a live peer already serves our port,
    so a losing race never registers the atexit cleanup that would delete the winner's file."""
    path = pidfile_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _port_in_use():
                return False
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        atexit.register(_remove_pidfile_if_ours)
        return True
    return False


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
    if not _claim_pidfile():
        sys.exit(0)
    serve(get_port())


if __name__ == '__main__':
    main()
