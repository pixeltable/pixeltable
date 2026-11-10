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

from pixeltable import exceptions as excs, exprs, func, type_system as ts
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
    # the name of the calling process's default time zone (Env.default_time_zone)
    time_zone: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    batch_args: list[list[Any]] | None = None
    batch_kwargs: dict[str, list[Any]] | None = None

    def encode(self, fn_call: exprs.FunctionCall) -> bytes:
        # self.signature gives no column type to a value passed through *args or **kwargs
        arg_types = [arg.col_type for arg in fn_call.args]
        kwarg_types = {name: arg.col_type for name, arg in fn_call.kwargs.items()}
        encode_media = proxy_protocol.encode_media_files
        wire: dict[str, Any] = {'signature': self.signature.as_dict(), 'time_zone': self.time_zone}
        if self.batch_args is not None:
            assert self.batch_kwargs is not None
            wire['batch_args'] = [
                [encode_media(val, arg_types[i]) for val in vals] for i, vals in enumerate(self.batch_args)
            ]
            wire['batch_kwargs'] = {
                name: [encode_media(val, kwarg_types[name]) for val in vals] for name, vals in self.batch_kwargs.items()
            }
        else:
            assert self.args is not None and self.kwargs is not None
            wire['args'] = [encode_media(val, arg_types[i]) for i, val in enumerate(self.args)]
            wire['kwargs'] = {name: encode_media(val, kwarg_types[name]) for name, val in self.kwargs.items()}
        return _encode_body(wire)

    @classmethod
    def decode(cls, body: bytes) -> Self:
        wire = _decode_body(body)
        return cls(
            func.Signature.from_dict(wire['signature']),
            wire['time_zone'],
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


# how long a call waits for the Server to start a container
_CONTAINER_WAIT_SECONDS = 300
# while calls are active, _monitor() sends a GET to the Server at least this often
_PROBE_SECONDS = 1.0
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
    """Sends calls to one UDF's Server, from one event loop.

    - A ModalScheduler creates one instance per query; the instance's asyncio tasks belong to the query's event loop.
    - call() POSTs each call's UdfCallRequest to the Server right away.
    - While calls are active, _monitor() sends a GET to the Server at least every _PROBE_SECONDS.
    - A GET answered with 503 means that the Server has no running container, so no POST sent before that GET can
      complete: Modal's proxy answers such a POST with 503 before reading its body and resets the stream.
    - _monitor() then cancels those POSTs. Their calls wait until a GET finds a running container, and then send their
      requests again.

    The GETs during POSTs and the cancellation work around a defect of httpcore 1.0.9: when the proxy resets the
    stream of a POST whose upload waits for HTTP/2 flow-control window, its _wait_for_outgoing_flow() keeps waiting
    until the read timeout instead of failing the request.

    TODO: once httpcore fails such a request right away:
    - remove self.posts and the cancellation
    - have _post() return None for the exception that httpcore raises
    - send GETs only while a call waits for a container
    """

    udf_path: str
    timeout: httpx.Timeout
    # the Server's URL; set by connect()
    url: str | None
    # clear if the last GET or POST found no running container
    container_running: asyncio.Event
    # the POSTs in flight, each with its send time (time.monotonic())
    posts: dict[asyncio.Task[httpx.Response], float]
    # the number of calls in call(), including the ones waiting for a container
    num_active_calls: int
    # runs _monitor() while num_active_calls > 0
    monitor: asyncio.Task[None] | None

    def __init__(self, udf_path: str, timeout_seconds: int) -> None:
        self.udf_path = udf_path
        self.timeout = httpx.Timeout(timeout_seconds, connect=_CONNECT_SECONDS)
        self.url = None
        self.container_running = asyncio.Event()
        self.container_running.set()
        self.posts = {}
        self.num_active_calls = 0
        self.monitor = None

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
        time_zone = Env.get().default_time_zone
        assert time_zone is not None
        request_body = UdfCallRequest(
            call_args.pxt_fn.signature,
            time_zone.key,
            call_args.args,
            call_args.kwargs,
            call_args.batch_args,
            call_args.batch_kwargs,
        ).encode(call_args.fn_call)
        http_client: httpx.AsyncClient = get_runtime().get_client('modal')
        start = time.monotonic()
        self.num_active_calls += 1
        if self.monitor is None:
            self.monitor = asyncio.create_task(self._monitor(http_client))

        try:
            while True:
                if not self.container_running.is_set():
                    await self._wait_for_container(start + _CONTAINER_WAIT_SECONDS)
                response = await self._post(http_client, request_body)
                if response is not None:
                    break
                # the next iteration waits for a GET that finds a container, or raises PROVIDER_ERROR at the deadline;
                # such GETs come at most every _PROBE_SECONDS, which limits how often a call retries after 503s
                self.container_running.clear()

        finally:
            self.num_active_calls -= 1
            if self.num_active_calls == 0:
                assert self.monitor is not None
                self.monitor.cancel()
                self.monitor = None

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

    async def _post(self, http_client: httpx.AsyncClient, request_body: bytes) -> httpx.Response | None:
        """POST request_body to the Server and return the response, or None if the Server had no running container."""
        assert self.url is not None
        post = asyncio.create_task(http_client.post(self.url, content=request_body, timeout=self.timeout))
        self.posts[post] = time.monotonic()
        try:
            # unlike awaiting post, asyncio.wait() returns normally when _monitor() cancels post
            await asyncio.wait([post])
        finally:
            del self.posts[post]
            # stops the POST if this call was cancelled; a no-op for a finished POST
            post.cancel()
        if post.cancelled():
            return None

        try:
            response = post.result()
        except httpx.LocalProtocolError:
            # the proxy reset the stream before reading the whole body, so the UDF didn't run; that reset raises
            # LocalProtocolError, not RemoteProtocolError (measured)
            return None
        except httpx.ReadTimeout as exc:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_TIMEOUT,
                f'Modal call of {self.udf_path} got no response within {self.timeout.read}s (timeout_seconds in the '
                'udf_resources setting)',
            ) from exc
        except httpx.TransportError as exc:
            # other transport errors, such as RemoteProtocolError, can also come after the UDF ran, so the call isn't
            # sent again
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR, f'Modal call of {self.udf_path} failed: {exc!r}'
            ) from exc

        # Modal answers 503 only when the Server has no running container; with every container busy, it queues the
        # request (measured)
        return None if response.status_code == 503 else response

    async def _monitor(self, http_client: httpx.AsyncClient) -> None:
        """Send GETs to the Server, set container_running from the answers, and cancel the POSTs that can't complete."""
        assert self.url is not None
        attempt = 0
        while True:
            sent = time.monotonic()
            # GETs during calls don't make Modal start more containers: a GET every 200 ms during sequential calls
            # started none (measured)
            try:
                status: int | None = (await http_client.get(self.url, timeout=_CONNECT_SECONDS)).status_code
            except httpx.HTTPError as exc:
                _logger.debug(f'GET of the Modal Server of {self.udf_path} failed: {exc!r}')
                status = None

            if status == 503:
                # there is no container running, we need to cancel the calls that were sent before our GET
                self.container_running.clear()
                stuck = [post for post, post_sent in self.posts.items() if post_sent < sent]
                for post in stuck:
                    post.cancel()
                if len(stuck) > 0:
                    _logger.debug(f'No Modal container of {self.udf_path} is running; cancelled {len(stuck)} POSTs')

                # after a 503, the first GETs come sooner, to catch a container that starts within seconds
                await asyncio.sleep(min(_PROBE_SECONDS, 0.1 * 2**attempt))
                attempt += 1
            else:
                if status is not None:
                    self.container_running.set()
                attempt = 0
                await asyncio.sleep(_PROBE_SECONDS)

    async def _wait_for_container(self, deadline: float) -> None:
        """Wait until _monitor() finds a running container, or raise PROVIDER_ERROR at deadline."""
        # TODO: detect a Server whose containers fail in ModalServer.start(), instead of waiting _CONTAINER_WAIT_SECONDS
        try:
            await asyncio.wait_for(self.container_running.wait(), max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'No Modal container of {self.udf_path} was ready within {_CONTAINER_WAIT_SECONDS}s',
            ) from None
