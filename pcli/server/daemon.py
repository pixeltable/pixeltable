"""`python -m pcli.server.daemon` - pcli FastAPI daemon."""

import atexit
import os

import uvicorn

from pcli.probe import get_port, pidfile_path
from pcli.server.app import create_app


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
    # access_log=False is the hard guarantee the daemon log won't grow per-request; log_config=None
    # also currently produces a silent access logger, but the explicit flag is robust to upstream changes.
    uvicorn.run(create_app(), host='127.0.0.1', port=get_port(), log_config=None, access_log=False)


if __name__ == '__main__':
    main()
