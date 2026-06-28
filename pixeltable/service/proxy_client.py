"""Client-side transport for the proxy service.

send_request() runs a remote catalog method and returns its result, re-raising any server-side error as
the identical pixeltable exception. Subclasses implement only the transport (_send): ProxyHttpClient over
HTTP, InProcessProxyClient in-process.
"""

from __future__ import annotations

import abc
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

import httpx

from pixeltable import exceptions as excs
from pixeltable.catalog.update_status import UpdateStatus
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.local_store import TempStore

from . import framing, proxy_dispatch, proxy_protocol
from .proxy_protocol import MediaUrlRef, ProxyRequest, ProxyResponse

if TYPE_CHECKING:
    from pixeltable.catalog.table_path import TablePathKey

# FileCache entries are keyed by URL; proxy-fetched media has no owning client column, so we tag it with a
# placeholder tbl_id/col_id (the cache key is the daemon media URL, which is stable per file).
_PROXY_MEDIA_TBL_ID = UUID(int=0)
_PROXY_MEDIA_COL_ID = 0


def _collect_media_refs(obj: Any, refs: set[str]) -> None:
    """Collect the refs of all MediaUrlRef sentinels reachable in a decoded response."""
    if isinstance(obj, MediaUrlRef):
        refs.add(obj.ref)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_media_refs(v, refs)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_media_refs(v, refs)
    elif isinstance(obj, UpdateStatus):
        for row in obj.rows or []:
            _collect_media_refs(row, refs)


def _replace_media_refs(obj: Any, resolved: dict[str, str]) -> Any:
    """Return obj with each MediaUrlRef replaced by its localized path from `resolved`."""
    if isinstance(obj, MediaUrlRef):
        return resolved[obj.ref]
    if isinstance(obj, dict):
        return {k: _replace_media_refs(v, resolved) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_media_refs(v, resolved) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_media_refs(v, resolved) for v in obj)
    if isinstance(obj, UpdateStatus):
        if obj.rows is not None:
            obj.rows[:] = [_replace_media_refs(row, resolved) for row in obj.rows]
        return obj
    return obj


class ProxyClient(abc.ABC):
    @abc.abstractmethod
    def _send(self, request_json: str, binary_parts: list[bytes]) -> tuple[str, list[bytes]]:
        """Transport: send the request (json head + binary parts), return the response (head + parts)."""

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
        binary_parts: list[bytes] = []
        request = ProxyRequest(
            class_name=class_name,
            method=method,
            args=proxy_protocol.serialize(args, binary_parts),
            path_key=None if path_key is None else path_key.as_dict(),
            snapshot_path_key=None if snapshot_key is None else snapshot_key.as_dict(),
        )
        response_json, response_parts = self._send(request.model_dump_json(), binary_parts)
        response = ProxyResponse.model_validate_json(response_json)
        response._binary_parts = response_parts
        return response

    def _localize_media(self, result: Any) -> Any:
        """Fetch any persisted media referenced by the result (MediaUrlRef) from the daemon into the local store.

        Default no-op; the HTTP transport overrides it. The client does this itself (rather than via a plan's
        cache-prefetch node) because it never executes plans.
        """
        return result

    def send_request(self, class_name: str, method: str, args: dict[str, Any]) -> Any:
        """Run a (path-less) catalog method and return its (deserialized) result."""
        response = self.send(class_name, method, args)
        if response.error is not None:
            raise excs.Error.from_dict(response.error)
        return self._localize_media(proxy_protocol.deserialize(response.result, response._binary_parts))

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
                refresh(proxy_protocol.deserialize(response.current_md, response._binary_parts))
            if response.error is not None:
                raise excs.Error.from_dict(response.error)
            if response.is_stale_md:
                continue  # server withheld a stale mutation; retry against the refreshed schema
            return self._localize_media(proxy_protocol.deserialize(response.result, response._binary_parts))


class InProcessProxyClient(ProxyClient):
    """In-process transport"""

    def _send(self, request_json: str, parts: list[bytes]) -> tuple[str, list[bytes]]:
        return proxy_dispatch.handle(request_json, parts)


class ProxyHttpClient(ProxyClient):
    """HTTP transport: POSTs requests to a proxy /rpc endpoint."""

    _endpoint: str
    _http: httpx.Client

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._http = httpx.Client(base_url=endpoint, timeout=httpx.Timeout(120.0))

    def _send(self, request_json: str, parts: list[bytes]) -> tuple[str, list[bytes]]:
        body = framing.encode_body(request_json.encode(), parts)
        response = self._http.post('/rpc', content=body, headers={'Content-Type': 'application/octet-stream'})
        response.raise_for_status()
        head, response_parts = framing.decode_body(response.content)
        return head.decode(), response_parts

    def _media_url(self, ref: str) -> str:
        return f'{self._endpoint}/media/{ref}'

    def _fetch_media(self, ref: str) -> bytes:
        response = self._http.get(f'/media/{ref}')
        response.raise_for_status()
        return response.content

    def _localize_media(self, result: Any) -> Any:
        refs: set[str] = set()
        _collect_media_refs(result, refs)
        if len(refs) == 0:
            return result

        cache = FileCache.get()
        resolved: dict[str, str] = {}
        to_fetch: list[str] = []
        for ref in refs:
            hit = cache.lookup(self._media_url(ref))
            if hit is not None:
                resolved[ref] = str(hit)
            else:
                to_fetch.append(ref)

        if len(to_fetch) > 0:
            # fetch the (WAN) bytes concurrently; FileCache bookkeeping stays on this thread (not thread-safe)
            with ThreadPoolExecutor(max_workers=min(16, len(to_fetch))) as executor:
                blobs = list(executor.map(self._fetch_media, to_fetch))
            for ref, data in zip(to_fetch, blobs):
                tmp = TempStore.create_path(extension=pathlib.Path(ref).suffix)
                with open(tmp, 'wb') as f:
                    f.write(data)
                resolved[ref] = str(cache.add(_PROXY_MEDIA_TBL_ID, _PROXY_MEDIA_COL_ID, self._media_url(ref), tmp))

        return _replace_media_refs(result, resolved)
