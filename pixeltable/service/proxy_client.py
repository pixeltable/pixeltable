"""Client-side transport for the proxy service.

send_request() runs a remote catalog method and returns its result, re-raising any server-side error as
the identical pixeltable exception. Subclasses implement only the transport (_send): ProxyHttpClient over
HTTP, InProcessProxyClient in-process.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable

import httpx

from pixeltable import exceptions as excs

from . import proxy_dispatch, proxy_protocol
from .proxy_protocol import ProxyRequest, ProxyResponse

if TYPE_CHECKING:
    from pixeltable.catalog.table_path import TablePathKey


class ProxyClient(abc.ABC):
    @abc.abstractmethod
    def _send(self, request_json: str) -> str:
        """Transport: send the request JSON to the server and return the response JSON."""

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
            args=proxy_protocol.serialize(args),
            path_key=None if path_key is None else path_key.as_dict(),
            snapshot_path_key=None if snapshot_key is None else snapshot_key.as_dict(),
        )
        return ProxyResponse.model_validate_json(self._send(request.model_dump_json()))

    def send_request(self, class_name: str, method: str, args: dict[str, Any]) -> Any:
        """Run a (path-less) catalog method and return its (deserialized) result."""
        response = self.send(class_name, method, args)
        if response.error is not None:
            raise excs.Error.from_dict(response.error)
        return proxy_protocol.deserialize(response.result)

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
                refresh(proxy_protocol.deserialize(response.current_md))
            if response.error is not None:
                raise excs.Error.from_dict(response.error)
            if response.is_stale_md:
                continue  # server withheld a stale mutation; retry against the refreshed schema
            return proxy_protocol.deserialize(response.result)


class InProcessProxyClient(ProxyClient):
    """In-process transport"""

    def _send(self, request_json: str) -> str:
        return proxy_dispatch.handle(request_json)


class ProxyHttpClient(ProxyClient):
    """HTTP transport: POSTs requests to a proxy /rpc endpoint."""

    _endpoint: str
    _http: httpx.Client

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._http = httpx.Client(base_url=endpoint, timeout=httpx.Timeout(120.0))

    def _send(self, request_json: str) -> str:
        response = self._http.post('/rpc', content=request_json, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        return response.text
