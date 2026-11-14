"""Client-side transport for the proxy service.

send_request() runs a remote catalog method and returns its result, re-raising any server-side error as
the identical pixeltable exception. Requests are POSTed to a proxy daemon's /rpc endpoint; the Transport
determines how those bytes reach the daemon (direct HTTP, or HTTPS through the Cloud gateway with a credential).
"""

from __future__ import annotations

import abc
import json
import logging
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from pixeltable import exceptions as excs
from pixeltable.catalog.catalog import Catalog, ProhibitedWriteError
from pixeltable.catalog.path import Path as CatalogPath
from pixeltable.catalog.update_status import UpdateStatus
from pixeltable.row import RowBatch
from pixeltable.utils.cloud_utils import (
    MAX_PRESIGNED_URL_KEYS,
    get_presigned_url_from_cloud,
    get_presigned_urls_from_cloud,
)
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.http import DOWNLOAD_USER_AGENT, fetch_url, retry_if_retryable_error
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectPath, StorageTarget

from . import proxy_protocol
from .management_client import Credential, is_refusal, refusal
from .proxy_protocol import (
    KEEP_ALIVE_S,
    InlinePartSink,
    MediaPath,
    PartSink,
    ProxyRequest,
    ProxyResponse,
    PxtArchivePartSink,
    decode_body,
    encode_body,
    iter_body_chunks,
)

if TYPE_CHECKING:
    from pixeltable.catalog.table_path import TablePathKey

_logger = logging.getLogger(__name__)

# FileCache entries are keyed by URL; proxy-fetched media has no owning client column, so we tag it with a
# placeholder tbl_id/col_id (the cache key is the daemon media URL, which is stable per file).
_PROXY_MEDIA_TBL_ID = UUID(int=0)
_PROXY_MEDIA_COL_ID = 0

# home-bucket media is read through URLs the control plane signs, never with credentials for the bucket
# every URL of a result is signed before its first download; one that expires before its download starts is signed
# again (_fetch_signed)
_SIGNED_URL_TTL_S = 900
_SIGNED_DOWNLOAD_TIMEOUT_S = 60.0


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

    @abc.abstractmethod
    def new_part_sink(self) -> PartSink:
        """Return a fresh sink for one logical request's media parts."""

    def close(self) -> None:
        """Release any transport resources."""


_MAX_IDLE_S = KEEP_ALIVE_S - 1.0  # so no connection is reused as the daemon closes it


class HttpTransport(Transport):
    """Direct HTTP to a reachable proxy daemon endpoint."""

    _endpoint: str
    _http: httpx.Client

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._http = httpx.Client(
            base_url=endpoint, timeout=httpx.Timeout(120.0), limits=httpx.Limits(keepalive_expiry=_MAX_IDLE_S)
        )
        # closes the pooled connections when the transport is collected without close(), as when the runtime holding
        # it is discarded
        weakref.finalize(self, self._http.close)

    def post(self, body: bytes) -> bytes:
        # an explicit Content-Length keeps the request unchunked on the wire
        headers = {'Content-Type': 'application/octet-stream', 'Content-Length': str(len(body))}
        response = self._http.post('/rpc', content=iter_body_chunks(body), headers=headers)
        response.raise_for_status()
        return response.content

    def fetch(self, url: str) -> Path:
        return fetch_url(url)

    def media_url(self, media_path: str) -> str:
        return f'{self._endpoint}/media/{media_path}'

    def new_part_sink(self) -> PartSink:
        return InlinePartSink()

    def close(self) -> None:
        self._http.close()


_CONNECT_TIMEOUT = 10.0
_RPC_TIMEOUT = 1800.0
_MAX_POOL_SIZE = 16  # matches the fetch_media download threadpool
# below the gateway's idle timeout, so no request goes out on a pooled connection the gateway has closed
_GATEWAY_IDLE_S = 60.0
_RETRY_MAX_DELAY = 90.0  # seconds of retrying a request the daemon never received

# Failures that leave the request undelivered: no connection, or the request could not be written.
_UNDELIVERED_EXC = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteError, httpx.WriteTimeout)
# Failures after the daemon received the request.
_UNANSWERED_EXC = (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError)


def _gateway_detail(response: httpx.Response) -> dict[str, Any] | None:
    """The gateway's error detail, {error_code, message, retryable, ...}; None for an answer of the daemon itself."""
    try:
        detail = response.json().get('detail')
    except ValueError:
        return None
    return detail if isinstance(detail, dict) else None


def _denied(exc: BaseException) -> PermissionError | None:
    """The OS refusing the socket, when that is what a transport error wraps: a firewall or sandbox that denies the
    socket denies it again on a retry."""
    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, PermissionError):
            return cause
        cause = cause.__cause__
    return None


def _unanswered(uri: str) -> excs.InternalError:
    return excs.InternalError(
        excs.ErrorCode.INTERNAL_ERROR,
        f'The database became unresponsive while handling this request: {uri}\n'
        'This may be caused by a query that was too large for the database to serve.\n'
        'If this happens repeatedly, try splitting large queries or inserts into smaller batches.',
    )


class GatewayTransport(Transport):
    """HTTPS to a hosted database's proxy daemon through the Cloud gateway, which checks the credential on every
    request (pooled, thread-safe)."""

    _org: str
    _db: str
    # read per request: a session renewed while this client lives reaches the next request
    _credential_cb: Callable[[], Credential]
    _endpoint: str
    _http: httpx.Client

    def __init__(self, org: str, db: str, credential_cb: Callable[[], Credential], origin: str):
        self._org = org
        self._db = db
        self._credential_cb = credential_cb
        # the gateway forwards /db/rpc and /db/media/{ref} to the daemon; media URLs are formed against it (see fetch())
        self._endpoint = f'{origin}/db'
        self._http = httpx.Client(
            base_url=self._endpoint,
            timeout=httpx.Timeout(_RPC_TIMEOUT, connect=_CONNECT_TIMEOUT),
            limits=httpx.Limits(max_connections=_MAX_POOL_SIZE, keepalive_expiry=_GATEWAY_IDLE_S),
        )
        # closes the pooled connections when the transport is collected without close(), as when the runtime holding
        # it is discarded
        weakref.finalize(self, self._http.close)

    def _request(self, method: str, path: str, body: bytes | None = None) -> bytes:
        """Issue one request with the current credential and return the raw body; a body is an encoded /rpc request.

        A failure that leaves the request undelivered (no connection, or the request could not be written) is
        retried with backoff, as is an answer that says to retry: a 429, a 5xx of the daemon itself, or a gateway
        answer marked retryable. A refused credential, a connection the OS refuses, a stopped database and any
        other 4xx are not retried.

        A failure after the daemon received the request is treated as a server crash and is not retried; retries
        in this scenario can inadvertently DOS the pod. A 502 or 504 of the gateway is one: the daemon may have
        received the request.
        """
        uri = f'pxt://{self._org}:{self._db}'

        @retry(
            retry=retry_if_exception_type(ConnectionError),
            wait=wait_exponential_jitter(initial=0.5, max=5.0),
            stop=stop_after_delay(_RETRY_MAX_DELAY),
            before_sleep=before_sleep_log(_logger, logging.DEBUG),
            reraise=True,
        )
        def _attempt() -> bytes:
            # before connecting: renewing a session is a round trip of its own, and a refused credential needs no socket
            credential = self._credential_cb()
            headers = {'Authorization': f'Bearer {credential.value}'}
            content = None
            if body is not None:
                # an explicit Content-Length keeps the request unchunked on the wire
                headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(len(body))})
                content = iter_body_chunks(body)
            try:
                response = self._http.request(method, path, content=content, headers=headers)
            except httpx.ProxyError as exc:
                # it refuses the same way again
                raise excs.ServiceUnavailableError(
                    excs.ErrorCode.DATABASE_UNAVAILABLE,
                    f'Cannot reach {uri}: the HTTPS proxy refused the connection ({exc}).',
                ) from exc
            except _UNDELIVERED_EXC as exc:
                denied = _denied(exc)
                if denied is not None:
                    raise denied from exc
                raise ConnectionError(f'{method} {path}: {exc}') from exc
            except _UNANSWERED_EXC as exc:
                raise _unanswered(uri) from exc
            if response.status_code == 200:
                return response.content
            detail = _gateway_detail(response)
            reason = (detail or {}).get('message') or response.reason_phrase or f'HTTP {response.status_code}'
            if response.headers.get('X-Pixeltable-Error-Code') in ('INVALID_STATE', 'USAGE_LIMIT'):
                # the database is stopped, and only its owner starts it
                raise excs.RequestError(excs.ErrorCode.INVALID_STATE, f'{uri}: {reason}')
            if is_refusal(response.status_code):
                raise refusal(response.status_code, reason, credential, f'connect to {uri}')
            # the gateway says whether to retry; an answer without its envelope is retried for a 429 or a 5xx of
            # the daemon itself, which is alive and answering
            if detail is not None:
                retryable = bool(detail.get('retryable'))
            else:
                retryable = response.status_code == 429 or response.status_code >= 500
            if retryable:
                try:
                    time.sleep(min(float(response.headers.get('Retry-After', 0)), _RETRY_MAX_DELAY))
                except ValueError:
                    pass
                raise ConnectionError(f'proxy {method} {path} error {response.status_code}: {reason}')
            raise _unanswered(uri)

        try:
            return _attempt()
        except ConnectionError as exc:
            # every try within the budget failed; every gateway host is <org>-<db>.<domain>, so allowing *.<domain>
            # covers every database
            domain = (urllib.parse.urlsplit(self._endpoint).hostname or '').partition('.')[2]
            raise excs.ServiceUnavailableError(
                excs.ErrorCode.DATABASE_UNAVAILABLE,
                f'Cannot reach {uri}: {exc}.\nIf this network limits outbound HTTPS, allow *.{domain} on port 443, '
                'or set HTTPS_PROXY to the proxy it uses.',
            ) from exc

    def post(self, body: bytes) -> bytes:
        return self._request('POST', '/rpc', body)

    def new_part_sink(self) -> PartSink:
        return PxtArchivePartSink(self._org, self._db)

    def media_url(self, media_path: str) -> str:
        return f'{self._endpoint}/media/{media_path}'

    def fetch(self, url: str) -> Path:
        # daemon media (/media/<ref>) needs the credential, which fetch_url() does not send; external s3/http URLs
        # fall back
        if not url.startswith(f'{self._endpoint}/media/'):
            return fetch_url(url)

        ref = url[len(self._endpoint) :]  # '/media/<ref>'
        tmp_path = TempStore.create_path(extension=Path(ref).suffix)
        tmp_path.write_bytes(self._request('GET', ref))
        return tmp_path

    def close(self) -> None:
        self._http.close()


def _sign_home_bucket_urls(urls: list[str]) -> dict[str, str]:
    """A signed https URL for each of urls that addresses a home bucket, in either spelling."""
    # (org, db, bucket) -> {key: [url]}: both spellings of an address name the same key, which is signed once
    by_bucket: defaultdict[tuple[str, str, str], dict[str, list[str]]] = defaultdict(dict)
    for url in urls:
        try:
            soa = ObjectPath.parse_object_storage_addr(url, allow_obj_name=True)
        except ValueError:
            continue  # the transport refuses it, as it did before
        if soa.storage_target == StorageTarget.PIXELTABLE_STORE:
            by_bucket[soa.account, soa.account_extension, soa.container].setdefault(soa.key, []).append(url)
    signed: dict[str, str] = {}
    for (org, db, bucket), urls_by_key in by_bucket.items():
        keys = list(urls_by_key)
        for i in range(0, len(keys), MAX_PRESIGNED_URL_KEYS):
            chunk = keys[i : i + MAX_PRESIGNED_URL_KEYS]
            url_by_key = _sign_keys(org, db, bucket, chunk)
            signed.update((url, url_by_key[key]) for key in chunk for url in urls_by_key[key])
    return signed


def _sign_keys(org: str, db: str, bucket: str, keys: list[str]) -> dict[str, str]:
    """A signed GET URL for each of at most MAX_PRESIGNED_URL_KEYS keys in a bucket of org/db, by key."""
    try:
        return get_presigned_urls_from_cloud(org, db, bucket, keys, expires_in=_SIGNED_URL_TTL_S)
    except excs.ExternalServiceError as e:
        # a control plane from before get_presigned_urls refuses the operation as a bad request
        if e.provider_http_status_code != 400:
            raise
    return {key: get_presigned_url_from_cloud(org, db, bucket, key, expiration=_SIGNED_URL_TTL_S) for key in keys}


@retry(
    retry=retry_if_retryable_error,
    wait=wait_exponential_jitter(initial=0.5, max=5.0),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request, timeout=_SIGNED_DOWNLOAD_TIMEOUT_S) as r, path.open('wb') as f:
        shutil.copyfileobj(r, f)


def _fetch_signed(url: str, signed_url: str) -> Path:
    """Download the media at url, a home-bucket address, from its signed URL into a temp file.

    fetch_url() cannot: it drops a URL's query string, which carries the signature.
    """
    try:
        return _fetch_signed_once(url, signed_url)
    except excs.ExternalServiceError as e:
        # an expired URL is refused like any other; the second refusal is the one raised
        if e.provider_http_status_code != 403:
            raise
    soa = ObjectPath.parse_object_storage_addr(url, allow_obj_name=True)
    return _fetch_signed_once(url, _sign_keys(soa.account, soa.account_extension, soa.container, [soa.key])[soa.key])


def _fetch_signed_once(url: str, signed_url: str) -> Path:
    path = TempStore.create_path(extension=Path(urllib.parse.urlparse(url).path).suffix)
    # the signed URL's query string is a credential, which a download error can hold (an HTTPError's url, a
    # ValueError's message): the error raised names url instead, and chains nothing
    try:
        _download(signed_url, path)
    except urllib.error.HTTPError as e:
        path.unlink(missing_ok=True)
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Failed to download {url}: HTTP {e.code}',
            provider='pixeltable_cloud',
            status_code=e.code,
        ) from None
    except Exception as e:
        path.unlink(missing_ok=True)
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR, f'Failed to download {url}: {type(e).__name__}', provider='pixeltable_cloud'
        ) from None
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


# Daemons from before the version fields put both numbers only in this sentence.
_LEGACY_PROTOCOL_MISMATCH_RE = re.compile(r'Unsupported proxy protocol version: (\d+) \(server expects (\d+)\)\Z')


class ProxyClient:
    """Talks to a proxy daemon: POSTs requests to its /rpc endpoint and localizes media results.

    The Transport determines how bytes reach the daemon (direct HTTP, or HTTPS through the Cloud gateway).
    """

    _transport: Transport

    def __init__(self, transport: Transport, catalog_uri: CatalogPath):
        self._transport = transport
        self._catalog_uri = catalog_uri

    @classmethod
    def local(cls, endpoint: str, db: str | None = None) -> ProxyClient:
        """Connect to a proxy daemon reachable directly over HTTP at endpoint."""
        return cls(HttpTransport(endpoint), CatalogPath(org='local', db=db))

    @classmethod
    def remote(cls, org: str, db: str, credential_cb: Callable[[], Credential], origin: str) -> ProxyClient:
        """Connect to a hosted database's proxy daemon through the Cloud gateway at origin."""
        return cls(GatewayTransport(org, db, credential_cb, origin), CatalogPath(org=org, db=db))

    def _prepare(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        """Serialize args for the wire, exactly once per logical request (media files are read, and for a
        hosted catalog uploaded, here; CAS retries must reuse the result rather than repeating that work)."""
        sink = self._transport.new_part_sink()
        return proxy_protocol.serialize_args(args, sink), sink.binary_parts

    def _post(
        self,
        class_name: str,
        method: str,
        wire_args: dict[str, Any],
        parts: list[bytes],
        *,
        path_key: TablePathKey | None = None,
        snapshot_key: TablePathKey | None = None,
    ) -> tuple[ProxyResponse, list[bytes]]:
        """POST one attempt of a prepared request and return the response head with its binary parts."""
        request = ProxyRequest(
            class_name=class_name,
            method=method,
            args=wire_args,
            path_key=None if path_key is None else path_key.as_dict(),
            snapshot_path_key=None if snapshot_key is None else snapshot_key.as_dict(),
            catalog_frozen=Catalog.is_frozen(),
        )
        body = encode_body(request.model_dump_json().encode(), parts)
        response_head, response_parts = decode_body(self._transport.post(body))
        return json.loads(response_head), response_parts

    def send(
        self,
        class_name: str,
        method: str,
        args: dict[str, Any],
        *,
        path_key: TablePathKey | None = None,
        snapshot_key: TablePathKey | None = None,
    ) -> tuple[ProxyResponse, list[bytes]]:
        """Run class_name.method(**args) on the server and return its head with its binary parts."""
        wire_args, parts = self._prepare(args)
        return self._post(class_name, method, wire_args, parts, path_key=path_key, snapshot_key=snapshot_key)

    def _validate_response(self, response: ProxyResponse) -> None:
        """Raise the server error. Fill in a short protocol-mismatch sentence from the version fields.

        A daemon from before those fields leaves the two numbers only in the sentence.
        """
        if response.get('prohibited_write', False):
            raise ProhibitedWriteError()
        error = response.get('error')
        if not isinstance(error, dict):
            if error is not None:
                raise excs.Error.from_dict(error)
            return
        message = error.get('message')
        legacy = _LEGACY_PROTOCOL_MISMATCH_RE.fullmatch(message) if isinstance(message, str) else None
        client_version = error.get('client_protocol_version')
        server_version = error.get('server_protocol_version')
        versions: tuple[int, int] | None
        if type(client_version) is int and type(server_version) is int:
            versions = (client_version, server_version)
        elif legacy is not None:
            versions = (int(legacy.group(1)), int(legacy.group(2)))
        else:
            versions = None
        if versions is not None and legacy is not None:
            error |= {'message': proxy_protocol.protocol_mismatch_message(*versions, self._catalog_uri)}
        raise excs.Error.from_dict(error)

    def send_request(self, class_name: str, method: str, args: dict[str, Any]) -> Any:
        """Run a (path-less) catalog method and return its (deserialized) result."""
        response, parts = self.send(class_name, method, args)
        self._validate_response(response)
        return self._localize_media(proxy_protocol.deserialize_value(response.get('result'), parts))

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
        wire_args, parts = self._prepare(args)
        while True:
            snapshot_key = get_snapshot_key()
            response, resp_parts = self._post(
                'Table', method, wire_args, parts, path_key=path_key, snapshot_key=snapshot_key
            )
            current_md = response.get('current_md')
            if current_md is not None:
                refresh(proxy_protocol.deserialize_value(current_md, resp_parts))
            self._validate_response(response)
            if response.get('is_stale_md', False):
                continue  # server withheld a stale mutation; retry against the refreshed schema
            return self._localize_media(proxy_protocol.deserialize_value(response.get('result'), resp_parts))

    def run_query(self, query_dict: dict, **extra: Any) -> Any:
        """Collect the rows of a query against the hosted catalog."""
        return self.send_request('Query', 'collect', {'query': query_dict, **extra})

    def _localize_media(self, result: Any) -> Any:
        """Resolve any MediaPath in the result to a fetchable daemon URL."""
        return _replace_media_paths(result, self._transport.media_url)

    def fetch_media(self, urls: list[str]) -> dict[str, str]:
        """Fetch each daemon/remote media URL into the local store, returning {url: local_path}.

        Home-bucket media comes through URLs the control plane signs, a call per MAX_PRESIGNED_URL_KEYS keys: the
        client never holds credentials for a bucket.
        """
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
            signed = _sign_home_bucket_urls(to_fetch)

            def fetch(url: str) -> Path:
                signed_url = signed.get(url)
                return self._transport.fetch(url) if signed_url is None else _fetch_signed(url, signed_url)

            # the transport handles every other supported scheme (the daemon's media URLs as well as external s3/http
            # media); fetch concurrently, but keep FileCache bookkeeping on this thread (not thread-safe)
            with ThreadPoolExecutor(max_workers=min(16, len(to_fetch))) as executor:
                tmp_paths = list(executor.map(fetch, to_fetch))
            for url, tmp in zip(to_fetch, tmp_paths):
                resolved[url] = str(cache.add(_PROXY_MEDIA_TBL_ID, _PROXY_MEDIA_COL_ID, url, tmp))

        return resolved

    def close(self) -> None:
        self._transport.close()
