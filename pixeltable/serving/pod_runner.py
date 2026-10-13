# This module intentionally omits from __future__ import annotations: the served application's route
# handlers carry annotations that FastAPI resolves at import time.

import argparse
import copy
import hashlib
import hmac
import logging
import os
import signal
import threading
import time
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

import uvicorn

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.service.db import report_instance_fingerprint
from pixeltable.serving._app import create_app, init_instrumentation, instrument_app
from pixeltable.serving._fastapi import FastAPIRouter
from pixeltable.utils.project import ProjectFingerprint

if TYPE_CHECKING:
    import fastapi


def _add_gateway_openapi_security(app: 'fastapi.FastAPI') -> None:
    original_openapi = app.openapi
    hosted_schema: dict[str, Any] | None = None

    def hosted_openapi() -> dict[str, Any]:
        nonlocal hosted_schema
        if hosted_schema is not None:
            return hosted_schema
        schema = copy.deepcopy(original_openapi())
        # X-api-key, not Bearer: the gateway reads it before Authorization, so it never collides with a
        # route's own Authorization-based scheme
        schemes = schema.setdefault('components', {}).setdefault('securitySchemes', {})
        schemes['PixeltableGatewayApiKey'] = {'type': 'apiKey', 'in': 'header', 'name': 'X-api-key'}
        for path, path_item in schema.get('paths', {}).items():
            if path == '/health':
                continue
            for method, operation in path_item.items():
                if method.lower() not in {'get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace'}:
                    continue
                app_security = operation.get('security', schema.get('security', [])) or [{}]
                operation['security'] = [{'PixeltableGatewayApiKey': [], **requirement} for requirement in app_security]
        hosted_schema = schema
        return schema

    app.openapi = hosted_openapi  # type: ignore[method-assign]


def _job_key(base_path: str, service_name: str) -> bytes:
    """Derive the MAC key for the service's job ids from the database credential, which all its pods share."""
    db_connect_str = Config.get().get_string_value('DB_CONNECT_STR')
    if db_connect_str is None:
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, 'a pod needs PIXELTABLE_DB_CONNECT_STR')
    return hmac.new(db_connect_str.encode(), f'pxt-job-id:{base_path}/{service_name}'.encode(), hashlib.sha256).digest()


class _DrainingServer(uvicorn.Server):
    """A uvicorn server that, on the first SIGTERM, waits for the routers' background jobs before shutting down.

    It keeps serving during the wait, so clients can still poll the jobs.
    """

    routers: list[FastAPIRouter]
    drain_s: float
    _drain_started: bool

    def __init__(self, config: uvicorn.Config, routers: list[FastAPIRouter], drain_s: float) -> None:
        super().__init__(config)
        self.routers = routers
        self.drain_s = drain_s
        self._drain_started = False

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        # a signal handler: it runs on the main thread, which may hold a router's _jobs_lock, so it takes no lock
        if sig == signal.SIGTERM and self.drain_s > 0 and not self._drain_started:
            self._drain_started = True
            threading.Thread(target=self._drain, args=(sig,), name='pxt-serve-drain', daemon=True).start()
            return
        super().handle_exit(sig, frame)

    def _drain(self, sig: int) -> None:
        if any(router._has_jobs() for router in self.routers):
            deadline = time.monotonic() + self.drain_s
            for router in self.routers:
                router._start_drain()
            for router in self.routers:
                router._wait_drained(deadline)
        # uvicorn's handle_exit() only records the signal and sets should_exit; both are safe on this thread
        super().handle_exit(sig, None)


def _serve(
    db_uri: str,
    app_file: str,
    service_name: str,
    base_path: str,
    project_dir: Path,
    fingerprint_file: Path,
    host: str,
    port: int,
    otel: bool,
) -> None:
    """Pod entry point: serve one service of the unpacked project and report what loaded."""
    if not project_dir.is_dir():
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'no project was unpacked at {project_dir}')
    if not fingerprint_file.is_file():
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{db_uri} has no project to serve {service_name} from')
    fingerprint = ProjectFingerprint.model_validate_json(fingerprint_file.read_text(encoding='utf-8'))
    # the unpacked project is this process's project root, so its modules and its database entry resolve
    Config.init(reinit=True, project_root=project_dir)

    if otel:
        # before the first Pixeltable operation, so that loading the file is traced too
        init_instrumentation()
    app, _, routers = create_app(str(project_dir / app_file), service_name, base_path)
    # the address of this pod for the other pods of the service
    instance_host = os.environ.get('PIXELTABLE_SERVE_INSTANCE_HOST')
    if instance_host is not None:
        instance_addr = f'[{instance_host}]:{port}' if ':' in instance_host else f'{instance_host}:{port}'
        key = _job_key(base_path, service_name)
        for router in routers:
            router._configure_job_forwarding(instance_addr, key)
    _add_gateway_openapi_security(app)
    if otel:
        instrument_app(app)
    report_instance_fingerprint(db_uri, service_name, fingerprint, base_path)

    log_level = logging.getLogger('pixeltable').getEffectiveLevel()
    # log_config=None keeps uvicorn from replacing the logging Env has already set up
    # root_path: the docs page fetches openapi.json by url, and a background route hands back a job url to
    # poll, so both need the whole prefix the gateway stripped, base_path included
    root_path = f'/{service_name}' if base_path == '' else f'/{base_path}/{service_name}'
    config = uvicorn.Config(
        app, host=host, port=port, log_level=log_level, log_config=None, root_path=root_path, proxy_headers=False
    )
    _DrainingServer(config, routers, float(os.environ.get('PIXELTABLE_SERVE_DRAIN_S', '0'))).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pixeltable.serving.pod_runner')
    parser.add_argument('--db', required=True, help='pxt://org:db, the database this pod belongs to')
    parser.add_argument('--app-file', required=True, help='path to the application file, from the project root')
    parser.add_argument('--name', required=True, help='the service to serve')
    parser.add_argument('--base-path', default='')
    parser.add_argument('--project-dir', type=Path, required=True, help='directory that holds the project contents')
    parser.add_argument('--project-fingerprint', type=Path, required=True, help='fingerprint of the project contents')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--otel', action='store_true')
    args = parser.parse_args()
    _serve(
        args.db,
        args.app_file,
        args.name,
        args.base_path,
        args.project_dir,
        args.project_fingerprint,
        args.host,
        args.port,
        args.otel,
    )
