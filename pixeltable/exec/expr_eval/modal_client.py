"""Client side of the UDFs in the udf_resources setting, which run on Modal Servers (pixeltable.serving.modal_server).

Contains the encoding of a call's request and response, and the client that sends a UDF's calls to its Server.
"""

import asyncio
import dataclasses
import json
import logging
import pickle
import time
from typing import Any

import httpx
from typing_extensions import Self

from pixeltable import exceptions as excs, func, type_system as ts
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.service import proxy_protocol

from .interfaces import FnCallArgs

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass(frozen=True)
class UdfCallRequest:
    """One call to a UDF, as sent to ModalServer.run().

    As in FnCallArgs, a call has either args and kwargs, for one row, or batch_args and batch_kwargs, with one value
    per row, for a batched UDF.
    """

    # ModalServer.run() calls the UDF's overload with this signature
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

    # one value per row of the request; None if error is set
    values: list[Any] | None
    # the exception raised by the UDF, which becomes the error of every row of the request
    error: Exception | None
    # how long the UDF ran in the container
    duration_seconds: float
    # Modal's task ID of the container that ran the call
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


# how long call() waits for the Server to start a container, while the Server answers 503
_CONTAINER_WAIT_SECONDS = 300
# the Servers' scaledown_window: Modal stops a container after it has been idle for at most this long
SCALEDOWN_SECONDS = 60
# time limit for opening a connection (TCP and TLS) to a Server; without it, a hung TLS handshake would wait for the
# call's full timeout
_CONNECT_SECONDS = 10


@register_client('modal', credential_param='proxy_secret')
def _(proxy_key: str, proxy_secret: str) -> httpx.AsyncClient:
    # with HTTP/2, concurrent calls to a Server share one connection
    # retries=2 retries a failed connection attempt up to twice; httpx never resends a request
    transport = httpx.AsyncHTTPTransport(http2=True, retries=2)
    return httpx.AsyncClient(transport=transport, headers={'Modal-Key': proxy_key, 'Modal-Secret': proxy_secret})


# the key of the process's ModalApp in Env.get_resource_pool_info()
_APP_REGISTRY_KEY = 'modal-app'


class ModalServerClient:
    """The process's client for one UDF's ModalServer."""

    udf_path: str
    timeout: httpx.Timeout
    # the Server's URL; set by connect()
    url: str | None
    # the Server is assumed to have a running container until this time.monotonic() value
    container_ready_until: float

    def __init__(self, udf_path: str, timeout_seconds: int) -> None:
        self.udf_path = udf_path
        self.timeout = httpx.Timeout(timeout_seconds, connect=_CONNECT_SECONDS)
        self.url = None
        self.container_ready_until = 0.0

    async def connect(self) -> None:
        """Look up the Server's URL."""
        if self.url is None:
            # get_client('modal') needs h2 for HTTP/2, and h2 is installed with modal; without modal, get_client() would
            # raise an ImportError for h2, which doesn't say to install modal
            Env.get().require_package('modal')
            # get_client() raises MISSING_CREDENTIALS if the proxy token isn't configured; checking that here avoids
            # deploying the App when every call would fail
            get_runtime().get_client('modal')
            from pixeltable.serving.modal_server import ModalApp  # modal is an optional package

            app = Env.get().get_resource_pool_info(_APP_REGISTRY_KEY, ModalApp)
            # server_url() deploys the App if this process hasn't yet; a deploy takes seconds to minutes and blocks in
            # Modal's client, so it runs in a thread
            self.url = await asyncio.to_thread(app.server_url, self.udf_path)

    async def call(self, call_args: FnCallArgs) -> UdfCallResponse:
        """Run call_args on the Server."""
        assert self.url is not None
        request_body = UdfCallRequest(
            call_args.pxt_fn.signature, call_args.args, call_args.kwargs, call_args.batch_args, call_args.batch_kwargs
        ).encode()
        http_client: httpx.AsyncClient = get_runtime().get_client('modal')
        start = time.monotonic()
        while True:
            # When the Server has no running container, Modal's proxy answers 503 before reading the request body and
            # then resets the stream; for a large body, httpx then raises LocalProtocolError or hangs until the timeout
            # instead of returning the 503. A GET has no body, so _wait_for_container() sends GETs until a container
            # is running. A busy container doesn't cause a 503: Modal queues the call.
            if time.monotonic() >= self.container_ready_until:
                await self._wait_for_container(http_client, start + _CONTAINER_WAIT_SECONDS)
            try:
                response = await http_client.post(self.url, content=request_body, timeout=self.timeout)
                if response.status_code != 503:
                    break
            except httpx.LocalProtocolError:
                # the stream reset described above: the Server didn't receive the whole request, so the UDF didn't run
                pass
            # no container was running after all
            self.container_ready_until = 0.0
        if response.status_code != 200:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Modal call of {self.udf_path} failed with status {response.status_code}: {response.text}',
            )
        # Modal keeps this call's container for up to SCALEDOWN_SECONDS after it becomes idle; assume half of that
        self.container_ready_until = time.monotonic() + SCALEDOWN_SECONDS / 2
        result = UdfCallResponse.decode(response.content)
        _logger.debug(
            f'Modal call of {self.udf_path}: {len(call_args.rows)} rows in {time.monotonic() - start:.2f}s, '
            f'UDF {result.duration_seconds:.2f}s, container {result.task_id}'
        )
        return result

    async def _wait_for_container(self, http_client: httpx.AsyncClient, deadline: float) -> None:
        """Send GETs to the Server until it answers with a status other than 503, which means a container is running.

        Raises PROVIDER_ERROR if no container is running by deadline.
        """
        assert self.url is not None
        attempt = 0
        # TODO: detect a Server whose containers fail in ModalServer.start(), instead of waiting _CONTAINER_WAIT_SECONDS
        while time.monotonic() < deadline:
            if (await http_client.get(self.url, timeout=self.timeout)).status_code != 503:
                return
            await asyncio.sleep(min(1.0, 0.1 * 2**attempt))
            attempt += 1
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'No Modal container of {self.udf_path} was ready within {_CONTAINER_WAIT_SECONDS}s',
        )
