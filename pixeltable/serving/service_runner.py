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

from pixeltable.config import Config
from pixeltable.serving._app import create_app_for_services, load_service_routers

from .service_deployment import ServiceDeployment


def _serve(app_file: str, service_name: str, base_path: str) -> None:
    """Service entrypoint: bind an ephemeral loopback port, record the deployment, and serve."""
    import uvicorn

    services = load_service_routers(app_file)
    app = create_app_for_services(services, app_file=app_file, base_path=base_path, service_name=service_name)
    spec = services[service_name].service_spec(service_name)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]

    deployment = ServiceDeployment.create(
        service_name=service_name, base_path=base_path, port=port, app_file=str(Path(app_file).resolve()), spec=spec
    )

    def _cleanup(*_: Any) -> None:
        deployment.remove()
        sys.exit(0)

    atexit.register(deployment.remove)
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
    args = parser.parse_args()
    Config.init(reinit=True, project_root=args.project_root)
    _serve(args.app_file, args.name, args.base_path)
