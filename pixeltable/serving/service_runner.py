# This module intentionally omits from __future__ import annotations: the served application's route
# handlers carry annotations that FastAPI resolves at import time.

import argparse
import atexit
import logging
import signal
import socket
import sys
from pathlib import Path
from typing import Any

import pixeltable.catalog as catalog
from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.serving._app import create_app, init_instrumentation, instrument_app
from pixeltable.utils.project import loaded_fingerprint

from .service_manager import ServiceManager


def _serve(app_file: str, service_name: str, base_path: str, otel: bool, port: int = 0) -> None:
    """Service entrypoint: bind a loopback port, record the service, and serve.

    port=0: the OS assigns a free one
    port>0: the process exits if it cannot be bound
    """
    import uvicorn

    if otel:
        # before the first Pixeltable operation, so that loading the file is traced too
        init_instrumentation()
    app, spec = create_app(app_file, service_name, base_path)
    if otel:
        instrument_app(app)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('127.0.0.1', port))
    except OSError as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot serve {service_name!r} on port {port}: {e.strerror}'
        ) from e
    port = sock.getsockname()[1]

    project_root = Config.get().project_root
    assert project_root is not None  # the app file was loaded from that root
    manager = ServiceManager()
    catalog_path = catalog.Path.parse(base_path, allow_empty_path=True)
    db_config = Config.get().get_database_config(catalog_path)
    record = manager.create(
        service_name=service_name,
        base_path=base_path,
        port=port,
        app_file=str(Path(app_file).resolve()),
        spec=spec,
        otel=otel,
        fingerprint=loaded_fingerprint(project_root, db_config),
    )

    def _cleanup(*_: Any) -> None:
        manager.remove(record)
        sys.exit(0)

    atexit.register(manager.remove, record)
    signal.signal(signal.SIGTERM, _cleanup)

    log_level = logging.getLogger('pixeltable').getEffectiveLevel()
    # log_config=None keeps uvicorn from replacing the logging Env has already set up
    uvicorn.Server(uvicorn.Config(app, log_level=log_level, log_config=None)).run(sockets=[sock])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pixeltable.serving.service_runner')
    parser.add_argument('--app-file', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--base-path', default='')
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--otel', action='store_true')
    parser.add_argument('--port', type=int, default=0, help='loopback port to serve on; 0 asks the OS for one')
    args = parser.parse_args()
    Config.init(reinit=True, project_root=args.project_root)
    _serve(args.app_file, args.name, args.base_path, args.otel, args.port)
