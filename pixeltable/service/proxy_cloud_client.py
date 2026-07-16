"""TLS tunnel client for connecting to cloud-hosted databases via the proxy daemon sidecar.

After a TLS handshake with the cloud endpoint, the client sends a PXT/1.0 CONNECT frame to
authenticate and identify the target database. The sidecar validates the API key and, on
success, forwards all subsequent bytes to the proxy daemon's HTTP server. This client then
issues HTTP POST /rpc requests over that tunnel.

Connection is persistent: the TLS socket is reused across multiple RPC calls. On any socket
error the connection is torn down and re-established on the next request (transparent reconnect).
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
_RPC_TIMEOUT = 1800.0


class _TunnelHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection backed by an already-established socket."""

    def __init__(self, host: str, sock: ssl.SSLSocket, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.sock = sock

    def connect(self) -> None:
        pass  # socket already set in __init__


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

            # Switch from the connect-phase timeout to the RPC timeout now that the
            # handshake is done. ssl_sock inherited _CONNECT_TIMEOUT from raw_sock;
            # without this the socket would time out on any RPC that takes > 30s.
            ssl_sock.settimeout(_RPC_TIMEOUT)
            self._ssl_sock = ssl_sock
            self._http_conn = _TunnelHTTPConnection(self._host, ssl_sock, timeout=_RPC_TIMEOUT)
        except Exception:
            (ssl_sock or raw_sock).close()
            raise

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

    def _post(self, body: bytes) -> bytes:
        """Transport only: POST the already-encoded body over the TLS tunnel and return the raw response bytes.

        The wire protocol (encode_body/decode_body) is handled by the base ProxyClient._send; this override
        exists solely to swap the transport to the reconnecting TLS tunnel.
        """
        for attempt in range(2):
            try:
                if self._ssl_sock is None:
                    self._connect()
                assert self._http_conn is not None
                self._http_conn.request('POST', '/rpc', body=body, headers={'Content-Type': 'application/octet-stream'})
                response = self._http_conn.getresponse()
                content = response.read()
                if response.status != 200:
                    raise ConnectionError(f'proxy RPC error {response.status}: {content.decode(errors="replace")}')
                return content
            except (ConnectionError, OSError, http.client.HTTPException, ssl.SSLError) as e:
                _logger.debug('proxy tunnel error (attempt %d): %s', attempt + 1, e)
                self._disconnect()
                if attempt == 1:
                    raise
        raise AssertionError('unreachable')
