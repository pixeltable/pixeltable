"""Client-side transport for the proxy service.

send_request() runs a remote catalog method and returns its result, re-raising any server-side error as
the identical pixeltable exception. Requests are POSTed to a proxy daemon's /rpc endpoint over HTTP.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

import httpx

from pixeltable import exceptions as excs
from pixeltable.catalog.update_status import UpdateStatus
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.http import fetch_url

from . import proxy_protocol
from .proxy_protocol import MediaPath, ProxyRequest, ProxyResponse, decode_body, encode_body

if TYPE_CHECKING:
    from pixeltable.catalog.table_path import TablePathKey

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
    return obj


class ProxyClient:
    """Talks to a proxy daemon over HTTP: POSTs requests to its /rpc endpoint and localizes media results."""

    _endpoint: str
    _http: httpx.Client

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._http = httpx.Client(base_url=endpoint, timeout=httpx.Timeout(120.0))

    def _send(self, request_json: str, parts: list[bytes]) -> tuple[str, list[bytes]]:
        """Transport: POST the request (json head + binary parts) to /rpc, return the response (head + parts)."""
        body = encode_body(request_json.encode(), parts)
        response = self._http.post('/rpc', content=body, headers={'Content-Type': 'application/octet-stream'})
        response.raise_for_status()
        head, response_parts = decode_body(response.content)
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

    def run_query(self, method: str, query_dict: dict, **extra: Any) -> Any:
        """Execute a Query method against the hosted catalog."""
        return self.send_request('Query', method, {'query': query_dict, **extra})

    def _media_url(self, media_path: str) -> str:
        return f'{self._endpoint}/media/{media_path}'

    def _localize_media(self, result: Any) -> Any:
        """Resolve any MediaPath in the result to a fetchable daemon URL."""
        return _replace_media_paths(result, self._media_url)

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
            # fetch_url() handles every supported scheme (the daemon's http media URLs as well as external s3/http
            # media); fetch concurrently, but keep FileCache bookkeeping on this thread (not thread-safe)
            with ThreadPoolExecutor(max_workers=min(16, len(to_fetch))) as executor:
                tmp_paths = list(executor.map(fetch_url, to_fetch))
            for url, tmp in zip(to_fetch, tmp_paths):
                resolved[url] = str(cache.add(_PROXY_MEDIA_TBL_ID, _PROXY_MEDIA_COL_ID, url, tmp))

        return resolved
