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
import ssl
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from uuid import UUID

import certifi
import httpx
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_result,
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
from pixeltable.utils.http import DOWNLOAD_USER_AGENT, fetch_url, retry_if_retryable_error, tcp_keepalive_options
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectPath, StorageTarget

from . import proxy_protocol
from .management_client import Credential, refusal, retry_after_seconds
from .proxy_protocol import (
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
    """Sends encoded RPC bytes and media between a ProxyClient and a proxy daemon, through a pooled httpx client."""

    _endpoint: str
    _http: httpx.Client

    def __init__(self, endpoint: str, http: httpx.Client):
        self._endpoint = endpoint
        self._http = http
        # closes the pooled connections when the transport is collected without close(), as when the runtime holding
        # it is discarded
        weakref.finalize(self, self._http.close)

    @abc.abstractmethod
    def post(self, body: bytes) -> bytes:
        """POST an encoded octet-stream body to /rpc and return the raw response bytes."""

    @abc.abstractmethod
    def fetch(self, url: str) -> Path:
        """Download one media URL to a temp file and return its path."""

    def media_url(self, media_path: str) -> str:
        """Build a fetchable URL for a media-dir-relative path served by the daemon."""
        return f'{self._endpoint}/media/{media_path}'

    @abc.abstractmethod
    def new_part_sink(self) -> PartSink:
        """Return a fresh sink for one logical request's media parts."""

    def close(self) -> None:
        """Close the pooled connections."""
        self._http.close()

    def _send(self, method: str, path: str, body: bytes | None, headers: dict[str, str]) -> httpx.Response:
        """Send one request to the daemon, with body as an octet stream."""
        content = None
        if body is not None:
            # an explicit Content-Length keeps the request unchunked on the wire
            headers = {**headers, 'Content-Type': 'application/octet-stream', 'Content-Length': str(len(body))}
            content = iter_body_chunks(body)
        return self._http.request(method, path, content=content, headers=headers)


# Below the 5 s keep-alive of daemons from earlier releases, which a newer client can still reach, so no connection is
# reused as the daemon closes it.
_MAX_IDLE_S = 4.0


class HttpTransport(Transport):
    """Direct HTTP to a reachable proxy daemon endpoint."""

    def __init__(self, endpoint: str):
        http = httpx.Client(
            base_url=endpoint, timeout=httpx.Timeout(120.0), limits=httpx.Limits(keepalive_expiry=_MAX_IDLE_S)
        )
        super().__init__(endpoint, http)

    def post(self, body: bytes) -> bytes:
        response = self._send('POST', '/rpc', body, {})
        response.raise_for_status()
        return response.content

    def fetch(self, url: str) -> Path:
        return fetch_url(url)

    def new_part_sink(self) -> PartSink:
        return InlinePartSink()


_CONNECT_TIMEOUT = 10.0
_RPC_TIMEOUT = 1800.0
_MAX_IDLE_CONNECTIONS = 16  # matches the fetch_media download threadpool
# below the 1800 s idle timeout of the gateway's load balancer, so that no request goes out on a pooled connection
# that the load balancer has closed
_MAX_GATEWAY_IDLE_S = 60.0
# seconds of retrying a request that did not reach the gateway, or an answer with a retryable error
_RETRY_MAX_DELAY = 90.0
# seconds of retrying while the gateway answers that the database is starting, which takes about 2 minutes
_START_MAX_DELAY = 300.0

# Failures that leave the request undelivered: no connection, or the request could not be written.
_UNDELIVERED_EXC = (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteError, httpx.WriteTimeout)
# Failures after the request may have reached the daemon, so a write may already be applied.
_UNANSWERED_EXC = (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError)


def _https_proxy(host: str) -> urllib.parse.SplitResult | None:
    """Return the proxy in HTTPS_PROXY for host, or None for a direct connection.

    NO_PROXY can exempt host. A proxy other than http:// or https://, such as socks5://, is not used, because httpx
    needs the optional socksio package for it.
    """
    url = urllib.request.getproxies_environment().get('https')
    # with a proxy in the environment, proxy_bypass() reads NO_PROXY there too, not in the system settings. Given
    # host:443, it matches an entry for the host, for one of its domains, or for the host and that port.
    if url is None or urllib.request.proxy_bypass(f'{host}:443'):
        return None
    try:
        # a proxy without a scheme is an http:// one, such as 'proxy.example.com:3128'
        proxy = urllib.parse.urlsplit(url if '://' in url else f'http://{url}')
        _ = proxy.port  # raises ValueError for a port outside 0-65535, which httpx would accept
    except ValueError:
        # urlsplit()'s error repeats the netloc, which can contain the proxy password
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, 'HTTPS_PROXY is not a valid URL.') from None
    return proxy if proxy.scheme in ('http', 'https') else None


def _gateway_detail(response: httpx.Response) -> dict[str, Any] | None:
    """Return the error in a gateway answer, which is excs.Error.to_dict() output; None for any other answer."""
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get('detail') if isinstance(body, dict) else None
    # a newer gateway can send an error_code that this client cannot rebuild
    if not isinstance(detail, dict) or detail.get('error_code') not in excs.ErrorCode.__members__:
        return None
    return detail


_E = TypeVar('_E', bound=BaseException)


def _find_cause(exc: BaseException, exc_type: type[_E]) -> _E | None:
    """Return the first exception of exc_type in the chain of exc, or None.

    httpcore raises its connect error 'from None', so the OS or TLS error is in __context__, not in __cause__.
    """
    cause = exc.__cause__ or exc.__context__
    while cause is not None:
        if isinstance(cause, exc_type):
            return cause
        cause = cause.__cause__ or cause.__context__
    return None


def _unanswered(uri: str) -> excs.InternalError:
    return excs.InternalError(
        excs.ErrorCode.INTERNAL_ERROR,
        f'The database became unresponsive while handling this request: {uri}\n'
        'This may be caused by a query that was too large for the database to serve.\n'
        'If this happens repeatedly, try splitting large queries or inserts into smaller batches.',
    )


def _last_answer_error(retry_state: RetryCallState) -> excs.Error | None:
    """Return the error in the answer to the last try; None when the try raised or succeeded."""
    outcome = retry_state.outcome
    if outcome is None or outcome.failed:
        return None
    result = outcome.result()
    return result if isinstance(result, excs.Error) else None


def _wait(retry_state: RetryCallState) -> float:
    """Wait for the Retry-After of the last answer, otherwise back off exponentially."""
    error = _last_answer_error(retry_state)
    # a Retry-After of 0 would resend at once for the whole budget
    if error is not None and error.retry_after is not None and error.retry_after > 0:
        return min(error.retry_after, _RETRY_MAX_DELAY)
    return wait_exponential_jitter(initial=0.5, max=5.0)(retry_state)


class _RetryBudget:
    """Decide when to stop retrying one request: after _RETRY_MAX_DELAY, or after _START_MAX_DELAY once the gateway
    has answered that the database is starting. Each request needs its own instance."""

    _starting: bool

    def __init__(self) -> None:
        self._starting = False

    def __call__(self, retry_state: RetryCallState) -> bool:
        error = _last_answer_error(retry_state)
        # a request to a database that was stopped for being idle starts it, and the gateway answers
        # DATABASE_UNAVAILABLE with a Retry-After until it is up. A failed connection in between keeps the longer wait.
        if (
            error is not None
            and error.error_code == excs.ErrorCode.DATABASE_UNAVAILABLE
            and error.retry_after is not None
        ):
            self._starting = True
        return stop_after_delay(_START_MAX_DELAY if self._starting else _RETRY_MAX_DELAY)(retry_state)


class GatewayTransport(Transport):
    """HTTPS to a hosted database's proxy daemon through the Cloud gateway, which checks the credential on every
    request (pooled, thread-safe)."""

    _org: str
    _db: str
    _uri: str
    # read per request: a session renewed while this client lives reaches the next request
    _credential_cb: Callable[[], Credential]
    # the gateway host and the wildcard for every database, such as
    # 'org-main.svc.pxt.run (or *.svc.pxt.run for every database)'
    _hosts_to_allow: str
    _proxy: urllib.parse.SplitResult | None

    def __init__(self, org: str, db: str, credential_cb: Callable[[], Credential], origin: str):
        self._org = org
        self._db = db
        self._uri = f'pxt://{org}:{db}'
        self._credential_cb = credential_cb
        host = urllib.parse.urlsplit(origin).hostname or ''
        # the gateway host is <org>-<db>.svc.<domain>, so its domain without the first label covers every database.
        # Some sandboxes allow only exact hosts, so the host comes first.
        self._hosts_to_allow = f'{host} (or *.{host.partition(".")[2]} for every database)'
        self._proxy = _https_proxy(host)
        # the gateway forwards /db/rpc and /db/media/{ref} to the daemon
        endpoint = f'{origin}/db'
        # the system store trusts a TLS-inspecting proxy whose CA is installed on this machine, and reads
        # SSL_CERT_FILE. certifi adds the CAs that httpx trusts by default.
        tls_context = ssl.create_default_context()
        tls_context.load_verify_locations(certifi.where())
        transport = httpx.HTTPTransport(
            proxy=None if self._proxy is None else self._proxy.geturl(),
            verify=tls_context,
            # no limit on open connections, so that a long request does not hold up the others
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=_MAX_IDLE_CONNECTIONS,
                keepalive_expiry=_MAX_GATEWAY_IDLE_S,
            ),
            # probes keep a NAT or firewall from dropping the connection during a long request, in which the daemon
            # sends nothing until it is done
            socket_options=tcp_keepalive_options(),
        )
        http = httpx.Client(
            base_url=endpoint, timeout=httpx.Timeout(_RPC_TIMEOUT, connect=_CONNECT_TIMEOUT), transport=transport
        )
        super().__init__(endpoint, http)

    def _request(self, method: str, path: str, body: bytes | None = None) -> bytes:
        """Issue one request with the current credential and return the raw body.

        body is an encoded /rpc request, or None for a GET. A failure that leaves the request undelivered (no
        connection, the request could not be written, or the HTTPS proxy answered 5xx or 429) is retried with backoff.
        So is an error the gateway marks retryable, or a 429 or 503 without the gateway's error body, after its
        Retry-After. Retries stop after _RETRY_MAX_DELAY, or after _START_MAX_DELAY once the gateway has answered that
        the database is starting. A refused credential, a socket the OS denies (PermissionError), a TLS certificate
        that does not verify, a proxy that refuses the connection and any other answer are not retried.

        A failure after the daemon received the request, a broken connection or a 5xx without the gateway's error
        body, is retried only for a GET, which changes nothing. Any other request may have crashed the daemon, and a
        write may already be applied.
        """
        idempotent = method == 'GET'

        @retry(
            retry=retry_if_exception_type(ConnectionError)
            | retry_if_result(lambda result: isinstance(result, excs.Error)),
            wait=_wait,
            stop=_RetryBudget(),
            before_sleep=before_sleep_log(_logger, logging.DEBUG),
            # the last try's content or error, or the exception it raised
            retry_error_callback=lambda retry_state: retry_state.outcome.result(),
        )
        def _attempt() -> bytes | excs.Error:
            """Return the content, or an error that allows a retry. Raise any other error."""
            # before connecting: renewing a session is a round trip of its own, and a refused credential needs no socket
            credential = self._credential_cb()
            try:
                response = self._send(method, path, body, credential.header())
            except httpx.ProxyError as exc:
                # the message is the proxy's answer to CONNECT, such as '502 Bad Gateway'. A 5xx or 429 can pass on a
                # retry, and a refusal cannot.
                status = str(exc).partition(' ')[0]
                if status == '429' or status.startswith('5'):
                    raise ConnectionError(f'the HTTPS proxy answered {exc}') from exc
                raise self._proxy_refusal(str(exc)) from exc
            except _UNDELIVERED_EXC as exc:
                denied = _find_cause(exc, PermissionError)
                if denied is not None:
                    # a firewall or sandbox that denies the socket denies it again on a retry
                    raise excs.ServiceUnavailableError(
                        excs.ErrorCode.DATABASE_UNAVAILABLE,
                        f'Cannot reach {self._uri}: the OS denied the connection ({denied}).\n{self._network_advice()}',
                    ) from exc
                cert_error = _find_cause(exc, ssl.SSLCertVerificationError)
                if cert_error is not None:
                    # a proxy or firewall that inspects TLS presents the same certificate on every try
                    raise excs.ServiceUnavailableError(
                        excs.ErrorCode.DATABASE_UNAVAILABLE,
                        f'Cannot reach {self._uri}: the TLS certificate did not verify ({cert_error}).\n'
                        f'A proxy or firewall on this network may inspect TLS: have it pass {self._hosts_to_allow} '
                        'through uninspected, or set SSL_CERT_FILE to a file with its CA certificate.',
                    ) from exc
                raise ConnectionError(f'{method} {path}: {exc}') from exc
            except _UNANSWERED_EXC as exc:
                if not idempotent:
                    raise _unanswered(self._uri) from exc
                return excs.ServiceUnavailableError(
                    excs.ErrorCode.DATABASE_UNAVAILABLE,
                    f'The connection to {self._uri} failed during {method} {path}: {exc}.',
                )
            if response.status_code == 200:
                return response.content
            error, allows_retry = self._answer_error(response, credential, idempotent)
            if not allows_retry:
                raise error
            return error

        try:
            result = _attempt()
        except ConnectionError as exc:
            # every try within the budget failed to reach the gateway
            raise excs.ServiceUnavailableError(
                excs.ErrorCode.DATABASE_UNAVAILABLE, f'Cannot reach {self._uri}: {exc}.\n{self._network_advice()}'
            ) from exc
        if isinstance(result, excs.Error):
            raise result
        return result

    def _network_advice(self) -> str:
        """Return what to allow on a network that limits outbound HTTPS."""
        if self._proxy is None:
            return (
                f'If this network limits outbound HTTPS, allow {self._hosts_to_allow} on port 443, or set HTTPS_PROXY '
                'to the proxy it uses.'
            )
        return f'If the HTTPS proxy limits outbound traffic, allow {self._hosts_to_allow} on port 443 in the proxy.'

    def _proxy_refusal(self, answer: str) -> excs.Error:
        """Return the error for an HTTPS proxy that refused the connection with answer, such as '403 Forbidden'."""
        if answer.startswith('407 '):
            # httpx raises ProxyError only for a connection through the proxy
            assert self._proxy is not None
            if self._proxy.username is None:
                fix = (
                    'Put the proxy credentials in HTTPS_PROXY, as '
                    f'{self._proxy.scheme}://user:password@{self._proxy.netloc}.'
                )
            else:
                fix = 'Check the proxy credentials in HTTPS_PROXY.'
            return excs.ServiceUnavailableError(
                excs.ErrorCode.DATABASE_UNAVAILABLE,
                f'Cannot reach {self._uri}: the HTTPS proxy requires authentication ({answer}).\n{fix}',
            )
        return excs.ServiceUnavailableError(
            excs.ErrorCode.DATABASE_UNAVAILABLE,
            f'Cannot reach {self._uri}: the HTTPS proxy refused the connection ({answer}).\n'
            f'Allow {self._hosts_to_allow} on port 443 in the proxy, or exempt it with NO_PROXY.',
        )

    def _answer_error(self, response: httpx.Response, sent: Credential, idempotent: bool) -> tuple[excs.Error, bool]:
        """Return the error for an answer other than 200, and whether the request can be sent again.

        An error the gateway marks retryable, or a 429 or 503 without the gateway's error body, allows a retry: each
        comes before the request reaches the daemon. Any other 5xx without that body allows a retry only for an
        idempotent request, because the daemon or a load balancer can send it after the daemon received the request.
        """
        status = response.status_code
        detail = _gateway_detail(response)
        if detail is not None:
            reason = detail.get('message', '')
            if status in (401, 403):
                # a refused credential reads like one the control plane refused, and says where the credential came
                # from. refusal() adds its own final period.
                return refusal(status, reason.rstrip('.'), sent, f'connect to {self._uri}'), False
            # every retryable error of the gateway comes with a Retry-After header, but not always with a retry_after
            error = excs.Error.from_dict(
                {**detail, 'message': f'{self._uri}: {reason}', 'retry_after': retry_after_seconds(response)}
            )
            return error, detail.get('retryable') is True
        if status in (429, 503) or (idempotent and status >= 500):
            # a load balancer answers 429 or 503 without passing the request on, and a GET can go again after any 5xx
            error = excs.ServiceUnavailableError(
                excs.ErrorCode.DATABASE_UNAVAILABLE,
                f'{self._uri} is unavailable (HTTP {status}).',
                retry_after=retry_after_seconds(response),
            )
            return error, True
        if status >= 500:
            return _unanswered(self._uri), False
        # a 4xx from something other than the gateway, such as a proxy or the daemon, or a 3xx, since httpx does not
        # follow redirects
        error = excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'A request to {self._uri} got an unexpected answer (HTTP {status}).',
            provider='pixeltable_cloud',
            status_code=status,
        )
        return error, False

    def post(self, body: bytes) -> bytes:
        return self._request('POST', '/rpc', body)

    def new_part_sink(self) -> PartSink:
        return PxtArchivePartSink(self._org, self._db)

    def fetch(self, url: str) -> Path:
        # daemon media (/media/<ref>) needs the credential, which fetch_url() does not send; external s3/http URLs
        # fall back
        if not url.startswith(f'{self._endpoint}/media/'):
            return fetch_url(url)

        ref = url[len(self._endpoint) :]  # '/media/<ref>'
        tmp_path = TempStore.create_path(extension=Path(ref).suffix)
        tmp_path.write_bytes(self._request('GET', ref))
        return tmp_path


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
