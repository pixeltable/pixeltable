"""TLS tunnel client for connecting to cloud-hosted databases via the proxy daemon sidecar.

After a TLS handshake with {org}-{db}.pxt.run:9000, the client sends a PXT/1.0 CONNECT frame
to authenticate and identify the target database. The sidecar validates the API key against
the gateway and, on success, forwards all subsequent bytes to the proxy daemon's HTTP server
(port 8000). This client then issues HTTP POST /rpc requests over that tunnel.

Connection is persistent: the TLS socket is reused across multiple RPC calls. On any socket
error the connection is torn down and re-established on the next request (transparent reconnect).

For dev clusters, set PIXELTABLE_CLOUD_HOST={org}-{db}.dev.pxt.run in the environment
(handled by runtime.py).
"""

from __future__ import annotations

import http.client
import logging
import socket
import ssl

from . import proxy_client

_logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9000
_CONNECT_TIMEOUT = 30.0
_RPC_TIMEOUT = 120.0


class ProxyCloudClient(proxy_client.ProxyClient):
    """HTTP-over-TLS-tunnel transport for a cloud-hosted database."""

    _org: str
    _db: str
    _api_key: str
    _host: str
    _port: int
    _ssl_sock: ssl.SSLSocket | None
    _http_conn: http.client.HTTPConnection | None

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
        self._ssl_sock = None
        self._http_conn = None

    def _connect(self) -> None:
        ctx = ssl.create_default_context()
        if self._no_verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        raw_sock = socket.create_connection((self._host, self._port), timeout=_CONNECT_TIMEOUT)
        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ssl_sock = ctx.wrap_socket(raw_sock, server_hostname=self._host)

        frame = f'PXT/1.0 CONNECT {self._org}/{self._db}\r\nAuthorization: Bearer {self._api_key}\r\n\r\n'
        ssl_sock.sendall(frame.encode())

        buf = b''
        while b'\r\n\r\n' not in buf:
            chunk = ssl_sock.recv(4096)
            if not chunk:
                ssl_sock.close()
                raise ConnectionError('Connection closed during PXT/1.0 handshake')
            buf += chunk

        first_line = buf.split(b'\r\n')[0].decode()
        if not first_line.startswith('PXT/1.0 200'):
            ssl_sock.close()
            raise PermissionError(f'PXT/1.0 handshake rejected: {first_line}')

        self._ssl_sock = ssl_sock
        # Wrap the already-connected socket in an HTTPConnection so we can use the
        # standard HTTP framing without reimplementing it.
        outer_sock = ssl_sock

        class _TunnelHTTP(http.client.HTTPConnection):
            def connect(inner_self) -> None:
                inner_self.sock = outer_sock

        conn = _TunnelHTTP(self._host, timeout=_RPC_TIMEOUT)
        conn.connect()
        self._http_conn = conn

    def _disconnect(self) -> None:
        if self._http_conn is not None:
            try:
                self._http_conn.close()
            except Exception:
                pass
            self._http_conn = None
        if self._ssl_sock is not None:
            try:
                self._ssl_sock.close()
            except Exception:
                pass
            self._ssl_sock = None

    def _send(self, request_json: str) -> str:
        for attempt in range(2):
            try:
                if self._ssl_sock is None:
                    self._connect()
                assert self._http_conn is not None
                self._http_conn.request(
                    'POST', '/rpc', body=request_json.encode(), headers={'Content-Type': 'application/json'}
                )
                response = self._http_conn.getresponse()
                body = response.read().decode()
                if response.status != 200:
                    raise ConnectionError(f'proxy RPC error {response.status}: {body}')
                return body
            except (ConnectionError, OSError, http.client.HTTPException, ssl.SSLError) as e:
                _logger.debug('proxy tunnel error (attempt %d): %s', attempt + 1, e)
                self._disconnect()
                if attempt == 1:
                    raise
        raise AssertionError('unreachable')
