"""Client side of the UDFs in the pixeltable.udf setting, which run on Modal Servers (pixeltable.serving.modal_server).

Contains the client that sends a UDF's calls to its Server through Modal's proxy.
"""

import asyncio
import logging
import time

import httpx

from pixeltable import exceptions as excs
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.service import udf_servers

from .interfaces import FnCallArgs

_logger = logging.getLogger('pixeltable')


# how long a call waits for the Server to start a container
_CONTAINER_WAIT_SECONDS = 300
# while calls are active, _monitor() sends a GET to the Server at least this often
_PROBE_SECONDS = 1.0
# time limit for opening a connection (TCP and TLS) to a Server; without it, a hung TLS handshake would wait for the
# call's full timeout
_CONNECT_SECONDS = 10


@register_client('modal', credential_param=None)
def _() -> httpx.AsyncClient:
    # with HTTP/2, concurrent calls to a Server share one connection
    # retries=2 retries a failed connection attempt up to twice; httpx never resends a request
    transport = httpx.AsyncHTTPTransport(http2=True, retries=2)
    return httpx.AsyncClient(transport=transport)


class ModalServerClient:
    """Sends calls to one UDF's Server, from one event loop.

    - A ModalScheduler creates one instance per execution of a plan; the instance's asyncio tasks belong to the
      plan's event loop.
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

    setting_path: str
    timeout: httpx.Timeout
    # set by connect()
    url: str | None
    headers: dict[str, str] | None
    # clear if the last GET or POST found no running container
    container_running: asyncio.Event
    # the POSTs in flight, each with its send time (time.monotonic())
    posts: dict[asyncio.Task[httpx.Response], float]
    # the number of calls in call(), including the ones waiting for a container
    num_active_calls: int
    # runs _monitor() while num_active_calls > 0
    monitor: asyncio.Task[None] | None

    def __init__(self, setting_path: str, timeout_seconds: int) -> None:
        self.setting_path = setting_path
        self.timeout = httpx.Timeout(timeout_seconds, connect=_CONNECT_SECONDS)
        self.url = None
        self.headers = None
        self.container_running = asyncio.Event()
        self.container_running.set()
        self.posts = {}
        self.num_active_calls = 0
        self.monitor = None

    async def connect(self) -> None:
        """Look up the Server's URL and the headers."""
        if self.url is not None:
            return
        endpoints = Env.get().udf_endpoints
        if endpoints is None:
            # a local database's lookup fingerprints the project and calls the control plane
            endpoints = await asyncio.to_thread(udf_servers.udf_endpoints)
        url = endpoints.server_urls.get(self.setting_path)
        if url is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{self.setting_path!r} is in the pixeltable.udf setting, but the database has no UDF server for '
                'it. Run `pxt db update` to deploy it.',
            )
        self.url = url
        self.headers = endpoints.headers

    async def call(self, call_args: FnCallArgs) -> udf_servers.UdfCallResponse:
        """Run call_args on the Server."""
        assert self.url is not None
        time_zone = Env.get().default_time_zone
        assert time_zone is not None
        request_body = udf_servers.UdfCallRequest(
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
            # endpoints from before a redeploy get 401 (another release's secret) or an error of the stopped servers
            Env.drop_udf_endpoints()
        if response.status_code == 401:
            # Modal's proxy answers 401 for a proxy token that it doesn't accept, and UdfServer.handle() for a wrong
            # secret; both are in UdfEndpoints.headers
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_AUTH_ERROR,
                f'The UDF server of {self.setting_path} refused the credentials of this process (HTTP 401). If the UDF '
                'servers were deployed again, run the operation again.',
            )
        if response.status_code != 200:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'The call to the UDF server of {self.setting_path} failed with status {response.status_code}: '
                f'{response.text}',
            )
        result = udf_servers.UdfCallResponse.decode(response.content)
        _logger.debug(
            f'Modal call of {self.setting_path}: {len(call_args.rows)} rows in {time.monotonic() - start:.2f}s, '
            f'UDF {result.duration_seconds:.2f}s, container {result.container_id}'
        )
        return result

    async def _post(self, http_client: httpx.AsyncClient, request_body: bytes) -> httpx.Response | None:
        """POST request_body to the Server and return the response, or None if the Server had no running container."""
        post = asyncio.create_task(
            http_client.post(self.url, content=request_body, headers=self.headers, timeout=self.timeout)
        )
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
                f'The call to the UDF server of {self.setting_path} got no response within {self.timeout.read}s '
                '(timeout_seconds in the pixeltable.udf setting)',
            ) from exc
        except httpx.TransportError as exc:
            # other transport errors, such as RemoteProtocolError, can also come after the UDF ran, so the call isn't
            # sent again
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR, f'The call to the UDF server of {self.setting_path} failed: {exc!r}'
            ) from exc

        # Modal answers 503 only when the Server has no running container; with every container busy, it queues the
        # request (measured)
        return None if response.status_code == 503 else response

    async def _monitor(self, http_client: httpx.AsyncClient) -> None:
        """Send GETs to the Server, set container_running from the answers, and cancel the POSTs that can't complete."""
        attempt = 0
        while True:
            sent = time.monotonic()
            # GETs during calls don't make Modal start more containers: a GET every 200 ms during sequential calls
            # started none (measured)
            try:
                response = await http_client.get(self.url, headers=self.headers, timeout=_CONNECT_SECONDS)
                status: int | None = response.status_code
            except httpx.HTTPError as exc:
                _logger.debug(f'GET of the Modal Server of {self.setting_path} failed: {exc!r}')
                status = None

            if status == 503:
                # there is no container running, we need to cancel the calls that were sent before our GET
                self.container_running.clear()
                stuck = [post for post, post_sent in self.posts.items() if post_sent < sent]
                for post in stuck:
                    post.cancel()
                if len(stuck) > 0:
                    _logger.debug(f'No Modal container of {self.setting_path} is running; cancelled {len(stuck)} POSTs')

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
        # TODO: detect a Server whose containers fail in UdfServer.start(), instead of waiting _CONTAINER_WAIT_SECONDS
        try:
            await asyncio.wait_for(self.container_running.wait(), max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'The UDF server of {self.setting_path} had no container ready within {_CONTAINER_WAIT_SECONDS}s',
            ) from None
