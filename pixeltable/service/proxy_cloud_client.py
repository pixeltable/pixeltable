"""TLS tunnel client for connecting to cloud-hosted databases via the proxy daemon sidecar.

Each connection does a TLS handshake with the cloud endpoint, then sends a PXT/1.0 CONNECT frame to
authenticate and identify the target database. The sidecar validates the API key and, on success,
forwards all subsequent bytes to the proxy daemon's HTTP server; this client then issues HTTP requests
(POST /rpc, GET /media/<ref>) over that tunnel.

Connections are pooled (`_TunnelPool`): a single TCP connection is serial for HTTP/1.1, so concurrent
SDK calls (and parallel media downloads) each borrow their own tunnel connection. Broken connections are
dropped and re-established on the next request (transparent reconnect). Transport only — the wire protocol
(encode_body/decode_body) and media caching stay in the base ProxyClient.
"""

from __future__ import annotations

import http.client
import logging
import socket
import ssl
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from . import proxy_client

_logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9000
_CONNECT_TIMEOUT = 30.0
_RPC_TIMEOUT = 1800.0
_MAX_POOL_SIZE = 16  # matches the base fetch_media download threadpool

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
    """Thread-safe pool of TLS + PXT/1.0 tunnel connections.

    borrow() hands out an idle connection or opens a new one; only on a clean return does it go back to the
    pool. On ANY exception it's closed and dropped — a broken transport can't be reused, and a connection
    whose request raised (e.g. a non-retryable 4xx -> RuntimeError) must not leak — then the exception
    propagates so the caller can retry on a fresh one where appropriate.
    """

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


class ProxyCloudClient(proxy_client.ProxyClient):
    """HTTP-over-TLS-tunnel transport for a cloud-hosted database (pooled, thread-safe)."""

    _org: str
    _db: str
    _api_key: str
    _host: str
    _port: int

    def __init__(
        self,
        org: str,
        db: str,
        api_key: str,
        host: str | None = None,
        port: int = _DEFAULT_PORT,
        no_verify: bool = False,
    ):
        self._org = org
        self._db = db
        self._api_key = api_key
        self._host = host or f'{org}-{db}.pxt.run'
        self._port = port
        self._no_verify = no_verify
        self._pool = _TunnelPool(self._connect_tunnel)
        # Base ProxyClient._media_url() builds media URLs as f'{self._endpoint}/media/<ref>'. Set _endpoint so
        # media results localize; those URLs are reachable only through the tunnel (see _fetch_url), not a
        # direct GET. We don't call super().__init__() because the base's httpx client is unused here — all
        # transport (RPC + media) goes through the tunnel pool.
        self._endpoint = f'https://{self._host}:{self._port}'

    def _connect_tunnel(self) -> http.client.HTTPConnection:
        """Open one tunnel connection: TCP + TLS + PXT/1.0 CONNECT handshake."""
        ctx = ssl.create_default_context()
        if self._no_verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
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
                # 5xx is transient (gateway/pod rollout) → retryable; anything else (4xx, media 404) is not.
                raise ConnectionError(msg) if response.status >= 500 else RuntimeError(msg)

        return _attempt()

    def _post(self, body: bytes) -> bytes:
        """RPC transport: POST the already-encoded body to /rpc over the tunnel (base _send handles the protocol)."""
        return self._request('POST', '/rpc', body=body, content_type='application/octet-stream')

    def _fetch_url(self, url: str) -> Path:
        """Media transport: pull daemon-served media (/media/<ref>) over the tunnel; delegate external URLs.

        The daemon serves persisted media at /media/<ref>, reachable only through the authenticated tunnel,
        so those go over the pool. Non-daemon URLs (external s3/http media) fall back to the base transport.
        FileCache bookkeeping stays in the base ProxyClient.fetch_media.
        """
        if not url.startswith(f'{self._endpoint}/media/'):
            return super()._fetch_url(url)
        from pixeltable.utils.local_store import TempStore

        ref = url[len(self._endpoint) :]  # '/media/<ref>'
        tmp_path = TempStore.create_path(extension=Path(ref).suffix)
        tmp_path.write_bytes(self._request('GET', ref))
        return tmp_path
