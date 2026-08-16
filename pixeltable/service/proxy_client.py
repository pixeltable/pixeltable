"""Client-side transport for the proxy service.

send_request() runs a remote catalog method and returns its result, re-raising any server-side error as
the identical pixeltable exception. Requests are POSTed to a proxy daemon's /rpc endpoint; the Transport
determines how those bytes reach the daemon (direct HTTP, or an authenticated TLS tunnel).
"""

from __future__ import annotations

import abc
import http.client
import logging
import socket
import ssl
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from pixeltable import exceptions as excs
from pixeltable.catalog.update_status import UpdateStatus
from pixeltable.row import RowBatch
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.http import fetch_url

from . import proxy_protocol
from .proxy_protocol import MediaPath, ProxyRequest, ProxyResponse, decode_body, encode_body

if TYPE_CHECKING:
    from pixeltable.catalog.table_path import TablePathKey

_logger = logging.getLogger(__name__)

# FileCache entries are keyed by URL; proxy-fetched media has no owning client column, so we tag it with a
# placeholder tbl_id/col_id (the cache key is the daemon media URL, which is stable per file).
_PROXY_MEDIA_TBL_ID = UUID(int=0)
_PROXY_MEDIA_COL_ID = 0


def _replace_media_paths(obj: Any, make_url: Callable[[str], str]) -> Any:
    """Return obj with each MediaPath replaced by its daemon url."""
    if isinstance(obj, MediaPath):
        return make_url(obj.path)
    if isinstance(obj, dict):
        return {k: _replace_media_paths(v, make_url) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_media_paths(v, make_url) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_media_paths(v, make_url) for v in obj)
    if isinstance(obj, UpdateStatus):
        if obj.rows is not None:
            obj.rows[:] = [_replace_media_paths(row, make_url) for row in obj.rows]
        return obj
    if isinstance(obj, RowBatch):
        return obj._map_values(lambda v: _replace_media_paths(v, make_url))
    return obj


class Transport(abc.ABC):
    """Sends encoded RPC bytes and media between a ProxyClient and a proxy daemon."""

    @abc.abstractmethod
    def post(self, body: bytes) -> bytes:
        """POST an encoded octet-stream body to /rpc and return the raw response bytes."""

    @abc.abstractmethod
    def fetch(self, url: str) -> Path:
        """Download one media URL to a temp file and return its path."""

    @abc.abstractmethod
    def media_url(self, media_path: str) -> str:
        """Build a fetchable URL for a media-dir-relative path served by the daemon."""

    def close(self) -> None:
        """Release any transport resources."""


class HttpTransport(Transport):
    """Direct HTTP to a reachable proxy daemon endpoint."""

    _endpoint: str
    _http: httpx.Client

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._http = httpx.Client(base_url=endpoint, timeout=httpx.Timeout(120.0))

    def post(self, body: bytes) -> bytes:
        response = self._http.post('/rpc', content=body, headers={'Content-Type': 'application/octet-stream'})
        response.raise_for_status()
        return response.content

    def fetch(self, url: str) -> Path:
        return fetch_url(url)

    def media_url(self, media_path: str) -> str:
        return f'{self._endpoint}/media/{media_path}'

    def close(self) -> None:
        self._http.close()


_CONNECT_TIMEOUT = 30.0
_RPC_TIMEOUT = 1800.0
_MAX_POOL_SIZE = 16  # matches the fetch_media download threadpool

# The server can restart and drop the connection mid-call; retry transient transport failures with backoff.
_TUNNEL_TRANSIENT_EXC = (ConnectionError, OSError, http.client.HTTPException, ssl.SSLError)
_TUNNEL_RETRY_MAX_DELAY = 90.0  # seconds; > _CONNECT_TIMEOUT so a hung handshake still leaves retry budget


class _TunnelHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection backed by an already-established socket."""

    def __init__(self, host: str, sock: ssl.SSLSocket, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.sock = sock

    def connect(self) -> None:
        pass  # socket already set in __init__


class _TunnelPool:
    """Thread-safe pool of TLS + PXT/1.0 tunnel connections."""

    def __init__(self, connect: Callable[[], http.client.HTTPConnection], max_size: int = _MAX_POOL_SIZE) -> None:
        self._connect = connect
        self._max = max_size
        self._lock = threading.Lock()
        self._idle: list[http.client.HTTPConnection] = []

    @contextmanager
    def borrow(self) -> Iterator[http.client.HTTPConnection]:
        with self._lock:
            conn = self._idle.pop() if self._idle else None
        conn = conn or self._connect()
        try:
            yield conn
        except BaseException:
            conn.close()  # never return a connection whose use raised (broken transport OR a raised HTTP error)
            raise
        else:
            with self._lock:
                if len(self._idle) < self._max:
                    self._idle.append(conn)
                else:
                    conn.close()

    def close(self) -> None:
        with self._lock:
            for conn in self._idle:
                try:
                    conn.close()
                except Exception:
                    pass
            self._idle.clear()


class TunnelTransport(Transport):
    """HTTP over an authenticated TLS tunnel to a proxy daemon behind the service sidecar (pooled, thread-safe)."""

    _org: str
    _db: str
    _api_key: str
    _host: str
    _port: int
    _endpoint: str
    _pool: _TunnelPool

    def __init__(self, org: str, db: str, api_key: str, host: str, port: int):
        self._org = org
        self._db = db
        self._api_key = api_key
        self._host = host
        self._port = port
        self._pool = _TunnelPool(self._connect_tunnel)
        # media URLs are formed against this endpoint; they are reachable only through the tunnel (see fetch())
        self._endpoint = f'https://{self._host}:{self._port}'

    def _connect_tunnel(self) -> http.client.HTTPConnection:
        """Open one tunnel connection: TCP + TLS + PXT/1.0 CONNECT handshake."""
        ctx = ssl.create_default_context()
        raw_sock = socket.create_connection((self._host, self._port), timeout=_CONNECT_TIMEOUT)
        ssl_sock: ssl.SSLSocket | None = None
        try:
            raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # TCP_KEEPIDLE is Linux; macOS uses TCP_KEEPALIVE for the same purpose
            keepidle = getattr(socket, 'TCP_KEEPIDLE', None) or getattr(socket, 'TCP_KEEPALIVE', None)
            if keepidle is not None:
                raw_sock.setsockopt(socket.IPPROTO_TCP, keepidle, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
            if hasattr(socket, 'TCP_KEEPCNT'):
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
            ssl_sock = ctx.wrap_socket(raw_sock, server_hostname=self._host)

            # the sidecar authenticates via the API key and routes the tunnel to org/db, then relays to the
            # proxy daemon's HTTP server; it answers 'PXT/1.0 200' on success (checked below)
            frame = f'PXT/1.0 CONNECT {self._org}/{self._db}\r\nAuthorization: Bearer {self._api_key}\r\n\r\n'
            ssl_sock.sendall(frame.encode())

            buf = b''
            while b'\r\n\r\n' not in buf:
                chunk = ssl_sock.recv(4096)
                if not chunk:
                    raise ConnectionError('Connection closed during PXT/1.0 handshake')
                buf += chunk

            first_line = buf.split(b'\r\n')[0].decode()
            if not first_line.startswith('PXT/1.0 200'):
                raise PermissionError(f'PXT/1.0 handshake rejected: {first_line}')

            # Switch from the connect-phase timeout to the RPC timeout now that the handshake is done;
            # otherwise the socket would time out on any request that takes longer than _CONNECT_TIMEOUT.
            ssl_sock.settimeout(_RPC_TIMEOUT)
            return _TunnelHTTPConnection(self._host, ssl_sock, timeout=_RPC_TIMEOUT)
        except Exception:
            (ssl_sock or raw_sock).close()
            raise

    def _request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
        """Borrow a tunnel connection, issue one request, return the raw body.

        Transient transport failures (the server can restart and drop the connection) are retried with backoff
        on a fresh connection; auth rejection (PermissionError) and non-5xx HTTP errors are not.
        """
        headers = {'Content-Type': content_type} if content_type else {}

        @retry(
            retry=retry_if_exception_type(_TUNNEL_TRANSIENT_EXC) & retry_if_not_exception_type(PermissionError),
            wait=wait_exponential_jitter(initial=0.5, max=5.0),
            stop=stop_after_delay(_TUNNEL_RETRY_MAX_DELAY),
            before_sleep=before_sleep_log(_logger, logging.DEBUG),
            reraise=True,
        )
        def _attempt() -> bytes:
            with self._pool.borrow() as conn:
                conn.request(method, path, body=body, headers=headers)
                response = conn.getresponse()
                content = response.read()
                if response.status == 200:
                    return content
                msg = f'proxy {method} {path} error {response.status}: {content.decode(errors="replace")}'
                # 5xx is transient (gateway/pod rollout) -> retryable; anything else (4xx, media 404) is not.
                raise ConnectionError(msg) if response.status >= 500 else RuntimeError(msg)

        return _attempt()

    def post(self, body: bytes) -> bytes:
        return self._request('POST', '/rpc', body=body, content_type='application/octet-stream')

    def media_url(self, media_path: str) -> str:
        return f'{self._endpoint}/media/{media_path}'

    def fetch(self, url: str) -> Path:
        # daemon media (/media/<ref>) is reachable only through the tunnel; external s3/http URLs fall back
        if not url.startswith(f'{self._endpoint}/media/'):
            return fetch_url(url)
        from pixeltable.utils.local_store import TempStore

        ref = url[len(self._endpoint) :]  # '/media/<ref>'
        tmp_path = TempStore.create_path(extension=Path(ref).suffix)
        tmp_path.write_bytes(self._request('GET', ref))
        return tmp_path

    def close(self) -> None:
        self._pool.close()


class ProxyClient:
    """Talks to a proxy daemon: POSTs requests to its /rpc endpoint and localizes media results.

    The Transport determines how bytes reach the daemon (direct HTTP, or an authenticated TLS tunnel).
    """

    _transport: Transport

    def __init__(self, transport: Transport):
        self._transport = transport

    @classmethod
    def local(cls, endpoint: str) -> ProxyClient:
        """Connect to a proxy daemon reachable directly over HTTP at endpoint."""
        return cls(HttpTransport(endpoint))

    @classmethod
    def remote(cls, org: str, db: str, api_key: str, host: str, port: int) -> ProxyClient:
        """Connect to the Pixeltable cloud service's proxy daemon over an authenticated TLS tunnel."""
        return cls(TunnelTransport(org, db, api_key, host=host, port=port))

    def _send(self, request_json: str, parts: list[bytes]) -> tuple[str, list[bytes]]:
        """Encode the request (json head + binary parts), POST it to /rpc, and decode the response."""
        body = encode_body(request_json.encode(), parts)
        head, response_parts = decode_body(self._transport.post(body))
        return head.decode(), response_parts

    def send(
        self,
        class_name: str,
        method: str,
        args: dict[str, Any],
        *,
        path_key: TablePathKey | None = None,
        snapshot_key: TablePathKey | None = None,
    ) -> ProxyResponse:
        """Run class_name.method(**args) on the server and return the raw response."""
        request = ProxyRequest(
            class_name=class_name,
            method=method,
            args=args,
            path_key=None if path_key is None else path_key.as_dict(),
            snapshot_path_key=None if snapshot_key is None else snapshot_key.as_dict(),
        )
        proxy_protocol.serialize_request(request)
        response_json, response_parts = self._send(request.model_dump_json(), request._binary_parts)
        response = ProxyResponse.model_validate_json(response_json)
        response._binary_parts = response_parts
        return response

    def send_request(self, class_name: str, method: str, args: dict[str, Any]) -> Any:
        """Run a (path-less) catalog method and return its (deserialized) result."""
        response = self.send(class_name, method, args)
        if response.error is not None:
            raise excs.Error.from_dict(response.error)
        return self._localize_media(proxy_protocol.deserialize_response(response, response.result))

    def dispatch_table_method(
        self,
        method: str,
        args: dict[str, Any],
        *,
        path_key: TablePathKey,
        get_snapshot_key: Callable[[], TablePathKey],
        refresh: Callable[[list], None],
    ) -> Any:
        """Run a Table method, refreshing the caller's local md from any current_md the server returns."""
        while True:
            snapshot_key = get_snapshot_key()
            response = self.send('Table', method, args, path_key=path_key, snapshot_key=snapshot_key)
            if response.current_md is not None:
                refresh(proxy_protocol.deserialize_response(response, response.current_md))
            if response.error is not None:
                raise excs.Error.from_dict(response.error)
            if response.is_stale_md:
                continue  # server withheld a stale mutation; retry against the refreshed schema
            return self._localize_media(proxy_protocol.deserialize_response(response, response.result))

    def run_query(self, method: str, query_dict: dict, **extra: Any) -> Any:
        """Execute a Query method against the hosted catalog."""
        return self.send_request('Query', method, {'query': query_dict, **extra})

    def _localize_media(self, result: Any) -> Any:
        """Resolve any MediaPath in the result to a fetchable daemon URL."""
        return _replace_media_paths(result, self._transport.media_url)

    def fetch_media(self, urls: list[str]) -> dict[str, str]:
        """Fetch each daemon/remote media URL into the local store, returning {url: local_path}."""
        cache = FileCache.get()
        resolved: dict[str, str] = {}
        to_fetch: list[str] = []
        for url in urls:
            hit = cache.lookup(url)
            if hit is not None:
                resolved[url] = str(hit)
            else:
                to_fetch.append(url)

        if len(to_fetch) > 0:
            # the transport handles every supported scheme (the daemon's media URLs as well as external s3/http
            # media); fetch concurrently, but keep FileCache bookkeeping on this thread (not thread-safe)
            with ThreadPoolExecutor(max_workers=min(16, len(to_fetch))) as executor:
                tmp_paths = list(executor.map(self._transport.fetch, to_fetch))
            for url, tmp in zip(to_fetch, tmp_paths):
                resolved[url] = str(cache.add(_PROXY_MEDIA_TBL_ID, _PROXY_MEDIA_COL_ID, url, tmp))

        return resolved

    def close(self) -> None:
        self._transport.close()
