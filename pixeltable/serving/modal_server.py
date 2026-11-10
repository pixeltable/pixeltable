"""The Modal App that runs the calls to the UDFs in the udf_resources setting, with one Modal Server per UDF."""

import asyncio
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import modal

from pixeltable import exceptions as excs, func
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.exec.expr_eval.modal_client import SCALEDOWN_SECONDS, UdfCallRequest, UdfCallResponse
from pixeltable.func.symbols import resolve_symbol
from pixeltable.utils.local_store import TempStore

_logger = logging.getLogger('pixeltable')

_PORT = 8000
# TODO: make the region a setting
# Modal's proxy and the Servers' containers run in this region; a container elsewhere adds a hop to every call
_REGION = 'us-east'
# Servers take no parameters, so the container gets the UDF path from this environment variable
_UDF_PATH_VAR = 'PXT_UDF_PATH'


class ModalServer:
    """Runs the calls to one UDF, one at a time, in a Modal container behind an HTTP server.

    The udf_resources config setting lists the UDFs that run on Modal, with the resources of their containers. ModalApp
    deploys one Modal Server per listed UDF, and each container of that Server runs an instance of this class, which
    serves the UDF's calls with hypercorn on _PORT. ModalServerClient sends each call as a POST of an encoded
    UdfCallRequest to Modal's proxy. The proxy checks the request's proxy token and forwards the request over HTTP/2
    to a container, which answers with an encoded UdfCallResponse.

    Why this class isn't using Modal functions (.remote()):
    - A Server call takes one round trip through Modal's proxy. A .remote() call takes two RPCs and moves an argument
      above 2 MB through Modal's blob storage.
    - With the Servers in us-east, an echo call took 74 ms at p50, against 153-165 ms with .remote(), and a call with
      a 1 MB argument took 223 ms, against 824 ms.
    - The calling process needs only a proxy token, which can call Servers but can't deploy Apps. .remote() needs a
      Modal API token.
    """

    udf_path: str
    overloads: list[func.CallableFunction]
    task_id: str | None
    lock: threading.Lock

    @modal.enter()
    def start(self) -> None:
        Env.init_without_db()
        self.udf_path = os.environ[_UDF_PATH_VAR]
        fn = resolve_symbol(self.udf_path)
        assert isinstance(fn, func.CallableFunction)
        self.overloads = fn._resolved_fns
        self.task_id = os.environ.get('MODAL_TASK_ID')
        self.lock = threading.Lock()
        # imported here because only the container needs hypercorn
        import hypercorn.asyncio
        import hypercorn.config

        config = hypercorn.config.Config()
        config.bind = [f'0.0.0.0:{_PORT}']
        # hypercorn runs in its own thread; with a shutdown_trigger, it doesn't install signal handlers, which Python
        # allows only in the main thread
        serve = hypercorn.asyncio.serve(self.handle, config, shutdown_trigger=asyncio.Event().wait)  # type: ignore[arg-type]
        threading.Thread(target=asyncio.run, args=(serve,), daemon=True).start()

    async def handle(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Answer a POST with the result of run() on its body."""
        if scope['type'] != 'http':
            return
        request_body = bytearray()
        while True:
            message = await receive()
            request_body += message.get('body', b'')
            if not message.get('more_body', False):
                break
        if scope['method'] != 'POST':
            # GETs come from Modal, which polls /metrics, and from ModalServerClient, which probes for a running
            # container
            status, body = 404, b''
        else:
            try:
                status, body = 200, await asyncio.to_thread(self.run, bytes(request_body))
            except Exception as exc:
                status, body = 500, f'{type(exc).__name__}: {exc}'.encode()
        await send(
            {'type': 'http.response.start', 'status': status, 'headers': [(b'content-length', str(len(body)).encode())]}
        )
        await send({'type': 'http.response.body', 'body': body})

    def run(self, request_body: bytes) -> bytes:
        """Run the UdfCallRequest in request_body and return the encoded UdfCallResponse."""
        with self.lock:
            try:
                request = UdfCallRequest.decode(request_body)
                fn = next((overload for overload in self.overloads if overload.signature == request.signature), None)
                if fn is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_STATE,
                        f"The Modal App's version of {self.udf_path!r} has no signature {request.signature}",
                    )
                start = time.monotonic()
                values: list[Any] | None = None
                error: Exception | None = None
                try:
                    if request.batch_args is not None:
                        assert request.batch_kwargs is not None
                        values = fn.exec_batch(request.batch_args, request.batch_kwargs)
                    else:
                        assert request.args is not None and request.kwargs is not None
                        values = [fn.py_fn(*request.args, **request.kwargs)]
                except Exception as exc:
                    error = exc
                response = UdfCallResponse(values, error, time.monotonic() - start, self.task_id)
                return response.encode(fn.signature.get_return_type())
            finally:
                # self.lock lets only one call run at a time, so the TempStore contains only this call's files
                TempStore.clear()


class ModalApp:
    """The current project's Modal App, with one Server per UDF in the udf_resources setting."""

    # keyed by UDF path
    server_urls: dict[str, str] | None
    lock: threading.Lock

    def __init__(self) -> None:
        self.server_urls = None
        self.lock = threading.Lock()

    def server_url(self, udf_path: str) -> str:
        """Return the URL of udf_path's Server, deploying the App at the first call."""
        with self.lock:
            if self.server_urls is None:
                self.server_urls = self._deploy()
            return self.server_urls[udf_path]

    def _deploy(self) -> dict[str, str]:
        """Deploy the App and return the URL of each UDF's Server, keyed by UDF path."""
        project_root = Config.get().project_root
        if project_root is None or not (project_root / 'uv.lock').is_file():
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                'Running UDFs on Modal requires a project with a uv.lock, which supplies the packages of the Modal '
                'image',
            )
        requirements = self._requirements(project_root)
        name = self._name(project_root)
        with tempfile.TemporaryDirectory() as tmp_dir:
            requirements_file = Path(tmp_dir) / 'requirements.txt'
            requirements_file.write_text(requirements, encoding='utf-8')
            image = (
                modal.Image.debian_slim(python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
                .apt_install('ffmpeg', 'git')
                # uv export lists every package with a pinned version; --no-deps installs exactly that list, because
                # resolving the dependencies again would ignore the project's override-dependencies
                .uv_pip_install(requirements=[str(requirements_file)], extra_options='--no-deps')
                .env({'HF_HOME': '/cache/hf'})
                .add_local_python_source('pixeltable', 'pixeltable_cli')
            )
            hf_cache = modal.Volume.from_name('pxt-hf-cache', create_if_missing=True)
            app = modal.App(name)
            servers: dict[str, modal.Server] = {}
            for udf_path, resources in Env.get().udf_resources().items():
                servers[udf_path] = app.server(
                    name=self._server_name(udf_path),
                    image=image,
                    serialized=True,
                    volumes={'/cache': hf_cache},
                    env={_UDF_PATH_VAR: udf_path},
                    port=_PORT,
                    h2_enabled=True,
                    routing_region=_REGION,
                    compute_region=_REGION,
                    # Modal's cpu is in physical cores, each with 2 vCPUs; UdfResources.cpu is in vCPUs
                    cpu=None if resources.cpu is None else resources.cpu / 2,
                    memory=resources.memory_mb,
                    gpu=resources.gpu,
                    # ModalServer runs one call at a time, so Modal starts another container when every one is busy
                    target_concurrency=1,
                    max_containers=resources.max_containers,
                    scaledown_window=SCALEDOWN_SECONDS,
                    # when Modal stops a container during a call, it waits up to exit_grace_period for the call to end
                    exit_grace_period=resources.timeout_seconds,
                )(ModalServer)
            _logger.info(f'Deploying Modal App {name!r}')
            start = time.monotonic()
            # 'recreate' stops the containers of the previous deploy; otherwise they would keep serving calls with
            # the previous deploy's pixeltable source until they scale down
            app.deploy(name=name, strategy='recreate')
        _logger.info(f'Deployed Modal App {name!r} in {time.monotonic() - start:.1f}s')
        return {udf_path: server.get_url() for udf_path, server in servers.items()}

    def _server_name(self, udf_path: str) -> str:
        return f'{udf_path.rsplit(".", 1)[-1]}-{hashlib.sha256(udf_path.encode()).hexdigest()[:8]}'

    def _requirements(self, project_root: Path) -> str:
        """Return the packages of the project's uv.lock that the image installs, in requirements format."""
        Env.get().require_binary('uv')
        export = subprocess.run(
            [
                'uv',
                'export',
                '--frozen',
                '--no-hashes',
                '--no-emit-workspace',  # leaves out the project's own packages, which are local paths
                '--no-emit-package',
                'pixeltable',  # the image gets pixeltable from add_local_python_source()
                '--no-emit-package',
                'modal',  # Modal installs its own client in the image
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if export.returncode != 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'uv export failed in {project_root}:\n{export.stderr}'
            )
        if re.search(r'^hypercorn==', export.stdout, re.MULTILINE) is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Running UDFs on Modal requires hypercorn in the uv.lock of {project_root}, for the HTTP server of '
                'the Modal containers; add it with `uv add hypercorn`',
            )
        return export.stdout

    def _name(self, project_root: Path) -> str:
        # the hash of project_root tells apart projects with the same directory name
        root_hash = hashlib.sha256(str(project_root).encode()).hexdigest()[:8]
        # Modal App names allow letters, digits, '-', '_' and '.'
        return f'pxt-udf-{re.sub(r"[^a-zA-Z0-9_.-]", "-", project_root.name)[:40]}-{root_hash}'
