"""The Modal App that runs the calls to the UDFs configured in udf_resources."""

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import modal
from typing_extensions import Self

from pixeltable import exceptions as excs, func, type_system as ts
from pixeltable.config import Config
from pixeltable.env import Env, register_client
from pixeltable.func.globals import resolve_symbol
from pixeltable.runtime import get_runtime
from pixeltable.service import proxy_protocol
from pixeltable.utils.local_store import TempStore

from .globals import FnCallArgs

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass(frozen=True)
class UdfCallRequest:
    """The arguments of a ModalServer.run() call, shaped as in FnCallArgs."""

    # selects the overload of the server's UDF
    signature: func.Signature
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    batch_args: list[list[Any]] | None = None
    batch_kwargs: dict[str, list[Any]] | None = None

    def encode(self) -> bytes:
        params = self.signature.parameters
        params_by_pos = self.signature.parameters_by_pos
        encode_media = proxy_protocol.encode_media_files
        wire: dict[str, Any] = {'signature': self.signature.as_dict()}
        if self.batch_args is not None:
            assert self.batch_kwargs is not None
            wire['batch_args'] = [
                [encode_media(val, params_by_pos[i].col_type) for val in vals] for i, vals in enumerate(self.batch_args)
            ]
            wire['batch_kwargs'] = {
                name: [encode_media(val, params[name].col_type) for val in vals]
                for name, vals in self.batch_kwargs.items()
            }
        else:
            assert self.args is not None and self.kwargs is not None
            wire['args'] = [encode_media(val, params_by_pos[i].col_type) for i, val in enumerate(self.args)]
            wire['kwargs'] = {name: encode_media(val, params[name].col_type) for name, val in self.kwargs.items()}
        return _encode_body(wire)

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        return cls(
            func.Signature.from_dict(wire['signature']),
            wire.get('args'),
            wire.get('kwargs'),
            wire.get('batch_args'),
            wire.get('batch_kwargs'),
        )


@dataclasses.dataclass(frozen=True)
class UdfCallResponse:
    """The result of a ModalServer.run() call."""

    # one per row; None if error is set
    values: list[Any] | None
    # the UDF's exception, which fails every row
    error: Exception | None
    # run time of the UDF
    duration_seconds: float
    # the Modal container that ran the call
    task_id: str | None

    def encode(self, return_type: ts.ColumnType) -> bytes:
        values = self.values
        if values is not None:
            values = [proxy_protocol.encode_media_files(val, return_type) for val in values]
        return _encode_body(
            {
                'values': values,
                'error': None if self.error is None else self._pickled_error(),
                'duration_seconds': self.duration_seconds,
                'task_id': self.task_id,
            }
        )

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        error = None if wire['error'] is None else pickle.loads(wire['error'])
        return cls(wire['values'], error, wire['duration_seconds'], wire['task_id'])

    def _pickled_error(self) -> bytes:
        """Pickle error, or a RuntimeError with its type and message if error does not survive pickling."""
        assert self.error is not None
        try:
            data = pickle.dumps(self.error)
            pickle.loads(data)
        except Exception:
            data = pickle.dumps(RuntimeError(f'{type(self.error).__name__}: {self.error}'))
        return data


def _encode_body(wire: dict[str, Any]) -> bytes:
    sink = proxy_protocol.InlinePartSink()
    head = json.dumps(proxy_protocol.serialize_args(wire, sink), separators=(',', ':')).encode()
    return proxy_protocol.encode_body(head, sink.binary_parts)


def _decode_body(body: bytes) -> dict[str, Any]:
    head, parts = proxy_protocol.decode_body(body)
    return proxy_protocol.deserialize_value(json.loads(head), parts)


_PORT = 8000
# TODO: make the region a setting
# Modal's proxy and the Servers' containers run in this region; a container elsewhere adds a hop to every call
_REGION = 'us-east'
# Servers take no parameters, so the container gets the UDF path from this environment variable
_UDF_PATH_VAR = 'PXT_UDF_PATH'
# how long a call waits for a container with room while the Server answers 503
_CONTAINER_WAIT_SECONDS = 300


class ModalServer:
    """Runs the calls to one UDF, one at a time, in a Modal container behind an HTTP server."""

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
        # only the container needs hypercorn
        import hypercorn.asyncio
        import hypercorn.config

        config = hypercorn.config.Config()
        config.bind = [f'0.0.0.0:{_PORT}']
        # with a shutdown_trigger, hypercorn installs no signal handlers, which work only in the main thread
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
            # Modal polls GET /metrics
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
                # the lock serializes the calls, so the temp store contains only this call's files
                TempStore.clear()


@register_client('modal', credential_param='proxy_secret')
def _(proxy_key: str, proxy_secret: str) -> httpx.AsyncClient:
    # HTTP/2 sends concurrent calls over one connection
    return httpx.AsyncClient(http2=True, headers={'Modal-Key': proxy_key, 'Modal-Secret': proxy_secret})


class ModalPool:
    """The process's handle on the Modal Server for one UDF."""

    udf_path: str
    timeout_seconds: int
    # set by connect()
    url: str | None

    def __init__(self, udf_path: str, timeout_seconds: int) -> None:
        self.udf_path = udf_path
        self.timeout_seconds = timeout_seconds
        self.url = None

    async def connect(self) -> None:
        """Look up the Server's URL, deploying the App at the process's first call."""
        if self.url is None:
            app = Env.get().get_resource_pool_info(_APP_REGISTRY_KEY, ModalApp)
            # a deploy takes seconds to minutes and calls Modal's blocking API
            app_name = await asyncio.to_thread(app.deploy)
            self.url = await modal.Server.from_name(app_name, _server_name(self.udf_path)).get_url.aio()

    async def call(self, call_args: FnCallArgs) -> UdfCallResponse:
        """Run call_args on the Server."""
        assert self.url is not None
        request_body = UdfCallRequest(
            call_args.pxt_fn.signature, call_args.args, call_args.kwargs, call_args.batch_args, call_args.batch_kwargs
        ).encode()
        client: httpx.AsyncClient = get_runtime().get_client('modal')
        start = time.monotonic()
        response = await client.post(self.url, content=request_body, timeout=self.timeout_seconds)
        attempt = 0
        # TODO: detect a Server whose containers fail in ModalServer.start(), instead of waiting _CONTAINER_WAIT_SECONDS
        while response.status_code == 503 and time.monotonic() - start < _CONTAINER_WAIT_SECONDS:
            await asyncio.sleep(min(1.0, 0.1 * 2**attempt))
            attempt += 1
            # probe with a GET, which has no body to upload again
            if (await client.get(self.url, timeout=self.timeout_seconds)).status_code != 503:
                response = await client.post(self.url, content=request_body, timeout=self.timeout_seconds)
        if response.status_code != 200:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Modal call of {self.udf_path} failed with status {response.status_code}: {response.text}',
            )
        result = UdfCallResponse.decode(response.content)
        _logger.debug(
            f'Modal call of {self.udf_path}: {len(call_args.rows)} rows in {time.monotonic() - start:.2f}s, '
            f'UDF {result.duration_seconds:.2f}s, container {result.task_id}'
        )
        return result


def _server_name(udf_path: str) -> str:
    """Return the name of the UDF's Server in the App."""
    return f'{udf_path.rsplit(".", 1)[-1]}-{hashlib.sha256(udf_path.encode()).hexdigest()[:8]}'


# ModalApp's key in Env.get_resource_pool_info()
_APP_REGISTRY_KEY = 'modal-app'


class ModalApp:
    """The current project's Modal App, with one Server per UDF in udf_resources."""

    # set by deploy()
    name: str | None
    # serializes deploy()
    lock: threading.Lock

    def __init__(self) -> None:
        self.name = None
        self.lock = threading.Lock()

    def deploy(self) -> str:
        """Deploy the App, unless this process already did, and return its name."""
        with self.lock:
            if self.name is None:
                self.name = self._deploy()
            return self.name

    def _deploy(self) -> str:
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
                # the export pins every package; resolving it again would drop the project's uv overrides
                .uv_pip_install(requirements=[str(requirements_file)], extra_options='--no-deps')
                .env({'HF_HOME': '/cache/hf'})
                .add_local_python_source('pixeltable', 'pixeltable_cli')
            )
            hf_cache = modal.Volume.from_name('pxt-hf-cache', create_if_missing=True)
            app = modal.App(name)
            for udf_path, resources in Env.get().udf_resources().items():
                app.server(
                    name=_server_name(udf_path),
                    image=image,
                    serialized=True,
                    volumes={'/cache': hf_cache},
                    env={_UDF_PATH_VAR: udf_path},
                    port=_PORT,
                    h2_enabled=True,
                    routing_region=_REGION,
                    compute_region=_REGION,
                    # Modal counts physical cores, each with 2 vCPUs
                    cpu=None if resources.cpu is None else resources.cpu / 2,
                    memory=resources.memory_mb,
                    gpu=resources.gpu,
                    # ModalServer runs one call at a time
                    target_concurrency=1,
                    # Modal can scale down a container during a call; this lets the call finish
                    exit_grace_period=resources.timeout_seconds,
                )(ModalServer)
            _logger.info(f'Deploying Modal App {name!r}')
            start = time.monotonic()
            # 'recreate' stops the previous deploy's containers, which would otherwise keep running its pixeltable
            # source
            app.deploy(name=name, strategy='recreate')
        _logger.info(f'Deployed Modal App {name!r} in {time.monotonic() - start:.1f}s')
        return name

    def _requirements(self, project_root: Path) -> str:
        """Return the packages of the project's uv.lock that the image installs, in requirements format."""
        Env.get().require_binary('uv')
        export = subprocess.run(
            [
                'uv',
                'export',
                '--frozen',
                '--no-hashes',
                '--no-emit-workspace',  # the project's own packages, which are local paths
                '--no-emit-package',
                'pixeltable',  # comes from add_local_python_source()
                '--no-emit-package',
                'modal',  # Modal adds its own client
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
        # tells apart projects with the same directory name
        root_hash = hashlib.sha256(str(project_root).encode()).hexdigest()[:8]
        # Modal App names allow letters, digits, '-', '_' and '.'
        return f'pxt-udf-{re.sub(r"[^a-zA-Z0-9_.-]", "-", project_root.name)[:40]}-{root_hash}'
