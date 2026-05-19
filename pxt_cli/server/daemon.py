"""`python -m pxt_cli.server.daemon` - pxt daemon entry point."""

import atexit
import os

from pxt_cli.probe import get_port, pidfile_path
from pxt_cli.server.app import serve


def _write_pidfile() -> None:
    path = pidfile_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    atexit.register(_remove_pidfile)


def _remove_pidfile() -> None:
    try:
        os.remove(pidfile_path())
    except OSError:
        pass


def main() -> None:
    _write_pidfile()
    serve(get_port())


if __name__ == '__main__':
    main()
