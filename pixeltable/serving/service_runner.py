"""Running a service declared in an application file, as a detached process per service.

The process serving a service writes its own registry record once it is listening, so the record and the
process appear together and the registry stays derived from the process table. Starting a service that is
already running and healthy returns the running one.
"""

# This module intentionally omits from __future__ import annotations: the served application's route
# handlers carry annotations that FastAPI resolves at import time.

import argparse
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from pixeltable import catalog, exceptions as excs
from pixeltable.config import Config
from pixeltable.serving._app import build_app_for_services, load_service_routers
from pixeltable.utils.process import pid_alive

from .service_registry import ServiceDeployment

_logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT = 60.0
_STOP_TIMEOUT = 10.0
_HEALTH_TIMEOUT = 2.0
_LOG_TAIL_LINES = 40


def log_path(name: str, base_path: str) -> Path:
    """The file a service's process writes its output to; the tree mirrors the registry's."""
    components = catalog.Path.parse(base_path, allow_empty_path=True).components
    return Config.get().home.joinpath('logs', 'services', *components, f'{name}.log')


def health_ok(endpoint: str) -> bool:
    """Whether the service at endpoint is listening and serving its application."""
    try:
        # every FastAPI application serves its schema, so this needs nothing of the service itself
        return httpx.get(f'{endpoint}/openapi.json', timeout=_HEALTH_TIMEOUT).status_code == 200
    except httpx.HTTPError:
        return False


def _tail_log(path: Path, n_lines: int = _LOG_TAIL_LINES) -> str:
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return ''
    return '\n'.join(lines[-n_lines:])


def start(app_file: str, name: str, base_path: str = '') -> ServiceDeployment:
    """Serve the named service from app_file with its models bound at base_path, and record it.

    Returns the deployment already running at that target, if there is one; nothing is restarted and
    app_file is not consulted in that case.

    Raises RequestError if the service cannot be served, and Error if its process never becomes healthy.
    """
    deployment = ServiceDeployment.read(name, base_path)
    if deployment is not None and health_ok(deployment.endpoint):
        return deployment

    # fail here, in the caller's process, on everything that can be detected without serving: an app file
    # that does not declare the service is a request error, not a process that dies in the background
    services = load_service_routers(app_file)
    if name not in services:
        declared = ', '.join(sorted(services))
        raise excs.NotFoundError(
            excs.ErrorCode.SERVICE_NOT_FOUND, f'{app_file} declares no service named {name!r}; it declares: {declared}'
        )

    path = log_path(name, base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        '-m',
        'pixeltable.serving.service_runner',
        '--app-file',
        app_file,
        '--name',
        name,
        '--base-path',
        base_path,
    ]
    # the service outlives this call, so it must not inherit our stdio: attached to a pipe it would block
    # the reader on EOF, and attached to a terminal it would write into that session. Its own session keeps
    # signals sent to the launching process away from it.
    with open(path, 'a', encoding='utf-8') as log_file:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        deployment = ServiceDeployment.read(name, base_path)
        if deployment is not None and health_ok(deployment.endpoint):
            return deployment
        if proc.poll() is not None:
            break
        time.sleep(0.25)

    msg = f'Service {name!r} failed to start within {_STARTUP_TIMEOUT:.0f}s'
    returncode = proc.poll()
    if returncode is None:
        msg += '; its process is still running but never reported healthy'
    else:
        msg += f'; its process exited with code {returncode}'
    tail = _tail_log(path)
    if tail != '':
        msg += f'\n--- service log tail ({path}) ---\n{tail}'
    raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, msg)


def stop(deployment: ServiceDeployment) -> None:
    """Terminate a service's process and remove its record."""
    pid: int | None = deployment.pid
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # POSIX raises ProcessLookupError once the process is gone; Windows raises PermissionError or OSError
        # from TerminateProcess for an already-exited process. In every case there is nothing left to wait on.
        pid = None
    if pid is not None:
        deadline = time.monotonic() + _STOP_TIMEOUT
        while time.monotonic() < deadline and pid_alive(pid):
            # reap promptly if the service is our own child, so its zombie isn't mistaken for a live process;
            # a no-op when another process launched it, in which case init reaps it
            if hasattr(os, 'WNOHANG'):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass
            time.sleep(0.05)
        if pid_alive(pid):
            # graceful shutdown overran the timeout; SIGKILL is absent on Windows, where os.kill() already
            # terminates unconditionally
            try:
                os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
            except (ProcessLookupError, PermissionError, OSError):
                pass
    deployment.remove()


def _serve(app_file: str, name: str, base_path: str) -> None:
    """Service entrypoint: bind an ephemeral loopback port, record the deployment, and serve."""
    import uvicorn

    # loaded once: importing the application file runs the user's module, side effects and all
    services = load_service_routers(app_file)
    app = build_app_for_services(services, app_file=app_file, base_path=base_path, name=name)
    spec = services[name].service_spec(name)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]

    deployment = ServiceDeployment(
        service_name=name,
        base_path=base_path,
        endpoint=f'http://127.0.0.1:{port}',
        pid=os.getpid(),
        created_at=time.time(),
        app_file=str(Path(app_file).resolve()),
        spec=spec,
    )
    deployment.write()

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
    args = parser.parse_args()
    _serve(args.app_file, args.name, args.base_path)
