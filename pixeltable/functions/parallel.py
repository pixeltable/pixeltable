"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) that wrap
[Parallel](https://parallel.ai/)'s web search and extraction APIs.

Without an API key, the UDFs call Parallel's free MCP endpoint, which needs `pip install mcp` and serves searches in
Fast mode under the service's anonymous rate limits. Specify `PARALLEL_API_KEY`, or `api_key` in the `parallel`
section of the Pixeltable config file, to call the Search and Extract REST APIs, which also accept the search and
extraction settings the free endpoint does not.
"""

import abc
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

import aiohttp

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import Env, register_client
from pixeltable.func.mcp import mcp_session
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import extract_retry_delay_from_headers

if TYPE_CHECKING:
    import mcp

_logger = logging.getLogger(__name__)

_REST_BASE_URL = 'https://api.parallel.ai'
# the endpoint Parallel recommends for new integrations; search.parallel.ai/mcp is the legacy standalone Search MCP
_MCP_URL = 'https://mcp.parallel.ai/mcp'
_REQUEST_TIMEOUT = 60.0
_USER_AGENT = f'Pixeltable/{pxt.__version__} (+https://github.com/pixeltable/pixeltable)'

SearchMode = Literal['turbo', 'fast', 'basic', 'advanced']

# How a failed response maps onto Pixeltable's error codes. The code decides whether the scheduler retries the row:
# it retries RATE_LIMITED, PROVIDER_OVERLOADED and an uncaused PROVIDER_ERROR, and nothing else.
_STATUS_ERROR_CODES: dict[int, excs.ErrorCode] = {
    400: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    401: excs.ErrorCode.PROVIDER_AUTH_ERROR,
    402: excs.ErrorCode.PROVIDER_AUTH_ERROR,
    403: excs.ErrorCode.PROVIDER_AUTH_ERROR,
    404: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    408: excs.ErrorCode.PROVIDER_TIMEOUT,
    422: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    429: excs.ErrorCode.RATE_LIMITED,
    503: excs.ErrorCode.PROVIDER_OVERLOADED,
    504: excs.ErrorCode.PROVIDER_TIMEOUT,
}


def _status_error(status: int, body: str = '', retry_after: float | None = None) -> excs.ExternalServiceError:
    message = f'Parallel returned HTTP {status}.'
    if len(body) > 0:
        message += f' {body[:500]}'
    if status in (401, 403):
        message += '\nCheck PARALLEL_API_KEY, or `api_key` in the `parallel` configuration section.'
    return excs.ExternalServiceError(
        _STATUS_ERROR_CODES.get(status, excs.ErrorCode.PROVIDER_ERROR),
        message,
        provider='parallel',
        status_code=status,
        retry_after=retry_after,
    )


def _missing_key_error(option: str) -> excs.AuthorizationError:
    return excs.AuthorizationError(
        excs.ErrorCode.MISSING_CREDENTIALS,
        f'`parallel` client not initialized with an API key: `{option}` requires one.\n'
        f'To fix this, specify the `PARALLEL_API_KEY` environment variable, or put `api_key` in the `parallel` '
        f'section of $PIXELTABLE_HOME/config.toml.',
    )


def _unwrapped(exc: BaseException) -> BaseException:
    """The transport error inside the ExceptionGroups anyio wraps it in, which computed-column errors report."""
    while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def _only_set(**fields: Any) -> dict[str, Any]:
    """The fields a caller gave a value, so that the service applies its own defaults to the rest."""
    return {name: value for name, value in fields.items() if value is not None}


class _ParallelClient(abc.ABC):
    """Calls Parallel's search and extraction over whichever transport the configured credentials allow.

    Both take the request body of the corresponding REST endpoint, and both return its documented response, so a
    column's contents do not depend on whether a key is configured.
    """

    @abc.abstractmethod
    async def search(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def extract(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class _ParallelRestClient(_ParallelClient):
    """Calls the Search and Extract REST APIs with the configured API key."""

    _request_headers: dict[str, str]
    _session: aiohttp.ClientSession | None

    def __init__(self, api_key: str):
        self._request_headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': _USER_AGENT,
        }
        # TODO: share the aiohttp session handling with `_JinaClient` and `_BflClient`; this is the third copy
        self._session = None  # defer session creation until we have a running event loop

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(base_url=_REST_BASE_URL)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post(self, endpoint: str, request: dict[str, Any]) -> dict[str, Any]:
        async with self._get_session().post(
            endpoint, json=request, headers=self._request_headers, timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        ) as resp:
            if resp.status != 200:
                raise _status_error(resp.status, await resp.text(), extract_retry_delay_from_headers(resp.headers))
            result = await resp.json()
            if not isinstance(result, dict):
                raise excs.ExternalServiceError(
                    excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    f'Parallel {endpoint} returned {type(result).__name__} rather than an object.',
                    provider='parallel',
                )
            return result

    async def search(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._post('/v1/search', request)

    async def extract(self, request: dict[str, Any]) -> dict[str, Any]:
        if 'allow_live_fetch' in request:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                '`allow_live_fetch` is only served by the free endpoint; with an API key, use `fetch_policy`.',
            )
        return await self._post('/v1/extract', request)


class _SessionTerminatedError(Exception):
    """The server no longer knows the session a call was made on, which one reconnect recovers from."""


class _ParallelMcpClient(_ParallelClient):
    """Calls Parallel's free MCP endpoint, which needs no account and serves searches in Fast mode."""

    _connection: asyncio.Task[None] | None
    _connected: 'asyncio.Future[mcp.ClientSession] | None'
    _disconnect: asyncio.Event | None
    _lock: asyncio.Lock

    def __init__(self) -> None:
        Env.get().require_package('mcp', [2])
        self._connection = None
        self._connected = None
        self._disconnect = None
        self._lock = asyncio.Lock()

    async def _connect(self, connected: 'asyncio.Future[mcp.ClientSession]', disconnect: asyncio.Event) -> None:
        """Hold one session open until `close()` or `_drop_session()` asks for it.

        The SDK's transport is an anyio task group, which has to be exited by the task that entered it, so the
        session lives in a task of its own rather than in whichever call opened it.
        """
        import httpx2

        try:
            async with (
                httpx2.AsyncClient(headers={'User-Agent': _USER_AGENT}, timeout=_REQUEST_TIMEOUT) as http_client,
                mcp_session(_MCP_URL, http_client=http_client, timeout=_REQUEST_TIMEOUT) as session,
            ):
                connected.set_result(session)
                await disconnect.wait()
        except Exception as exc:
            # a session that drops after it was handed out surfaces through the calls that are using it
            if connected.done():
                _logger.info(f'Parallel MCP session ended: {exc}')
            else:
                connected.set_exception(_unwrapped(exc))

    async def _get_session(self) -> 'mcp.ClientSession':
        async with self._lock:
            if self._connection is None or self._connection.done():
                connected: asyncio.Future[mcp.ClientSession] = asyncio.get_running_loop().create_future()
                disconnect = asyncio.Event()
                self._connected, self._disconnect = connected, disconnect
                self._connection = asyncio.create_task(self._connect(connected, disconnect))
            assert self._connected is not None
            return await self._connected

    async def _drop_session(self, session: 'mcp.ClientSession') -> None:
        """End the connection that produced `session`, leaving a successor another call opened in its place."""
        async with self._lock:
            if self._connected is None or not self._connected.done() or self._connected.result() is not session:
                return
            connection, self._connection = self._connection, None
            assert self._disconnect is not None
            self._disconnect.set()
        if connection is not None:
            await connection

    async def close(self) -> None:
        async with self._lock:
            connection, self._connection = self._connection, None
            if self._disconnect is not None:
                self._disconnect.set()
        if connection is not None:
            await connection

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = await self._get_session()
        try:
            return await self._invoke(session, tool, arguments)
        except _SessionTerminatedError:
            # The server no longer knows this session (idle expiry, a redeploy). The SDK fails only this request and
            # leaves the transport running, so without a reconnect every later call would reuse the dead session.
            await self._drop_session(session)
        try:
            return await self._invoke(await self._get_session(), tool, arguments)
        except _SessionTerminatedError as exc:
            # a session the server forgets as soon as it hands it out is not one another reconnect would fix
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_BAD_REQUEST, f'Parallel {tool} failed: {exc}', provider='parallel'
            ) from None

    async def _invoke(self, session: 'mcp.ClientSession', tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        import mcp

        try:
            result = await session.call_tool(tool, arguments)
        except mcp.MCPError as exc:
            raise self._call_error(tool, exc) from None
        except Exception as exc:
            raise _unwrapped(exc) from None

        text = '\n'.join(item.text for item in result.content if isinstance(item, mcp.types.TextContent))
        if result.is_error:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_BAD_REQUEST, f'Parallel {tool} failed: {text}', provider='parallel'
            )
        if isinstance(result.structured_content, dict):
            return result.structured_content
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                f'Parallel {tool} returned an unusable result: {text[:500]}',
                provider='parallel',
            )
        return parsed

    @classmethod
    def _call_error(cls, tool: str, exc: 'mcp.MCPError') -> Exception:
        """What a failed call means, as the SDK reports it: it keeps neither the status nor the body.

        A failed response that carries no JSON-RPC error body arrives as `INTERNAL_ERROR`, whatever its status, so a
        rate-limited or overloaded endpoint reaches the scheduler as a `PROVIDER_ERROR` it retries. Any other code is
        the server's own JSON-RPC error, which rerunning the row would only repeat.
        """
        import mcp

        if exc.message == 'Session terminated':
            return _SessionTerminatedError(exc.message)
        error_code = (
            excs.ErrorCode.PROVIDER_ERROR
            if exc.code == mcp.types.INTERNAL_ERROR
            else excs.ErrorCode.PROVIDER_BAD_REQUEST
        )
        return excs.ExternalServiceError(error_code, f'Parallel {tool} failed: {exc.message}', provider='parallel')

    async def search(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._call('web_search', self._tool_arguments(request))

    async def extract(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._call('web_fetch', self._tool_arguments(request))

    @classmethod
    def _tool_arguments(cls, request: dict[str, Any]) -> dict[str, Any]:
        """The free endpoint's tool arguments for a REST request, rejecting the fields that need a key.

        `web_search` and `web_fetch` take the request's top-level fields, plus `full_content` lifted out of
        `advanced_settings`. They serve Fast mode only, and no other advanced setting.
        """
        mode = request.get('mode', 'fast')
        if mode != 'fast':
            raise _missing_key_error(f'mode={mode!r}')
        arguments = {name: value for name, value in request.items() if name not in ('mode', 'advanced_settings')}
        for name, value in request.get('advanced_settings', {}).items():
            if name != 'full_content':
                raise _missing_key_error(name)
            arguments[name] = value
        return arguments


@register_client('parallel', credential_param='api_key')
def _(api_key: str | None = None) -> _ParallelClient:
    return _ParallelRestClient(api_key) if api_key is not None else _ParallelMcpClient()


def _client() -> _ParallelClient:
    return get_runtime().get_client('parallel')


class ParallelSourcePolicy(TypedDict):
    """Which sources a search may draw on."""

    include_domains: NotRequired[list[str]]
    exclude_domains: NotRequired[list[str]]
    after_date: NotRequired[str]


class ParallelFetchPolicy(TypedDict):
    """How fresh the page contents behind a request have to be."""

    max_age_seconds: NotRequired[int]
    timeout_seconds: NotRequired[float]
    disable_cache_fallback: NotRequired[bool]


class ParallelWarning(TypedDict):
    """A non-fatal adjustment Parallel made to a request."""

    type: str
    message: str
    detail: NotRequired[dict[str, Any] | None]


class ParallelUsage(TypedDict):
    name: str
    count: int


class ParallelSearchResult(TypedDict):
    url: str
    title: str | None
    publish_date: str | None
    excerpts: list[str]


class ParallelSearchResponse(TypedDict):
    search_id: str
    results: list[ParallelSearchResult]
    warnings: list[ParallelWarning] | None
    usage: NotRequired[list[ParallelUsage] | None]
    session_id: str


@pxt.udf(is_deterministic=False, resource_pool='request-rate:parallel')
async def search(
    objective: str,
    *,
    search_queries: list[str] | None = None,
    mode: SearchMode = 'fast',
    max_results: int | None = None,
    source_policy: ParallelSourcePolicy | None = None,
    fetch_policy: ParallelFetchPolicy | None = None,
    session_id: str | None = None,
) -> ParallelSearchResponse:
    """Search the web and return source URLs, titles and relevant excerpts.

    Equivalent to the Parallel Search API endpoint.
    For additional details, see: <https://docs.parallel.ai/search/search-quickstart>

    Without an API key, the free endpoint serves `fast` searches; `mode`, `max_results`, `source_policy` and
    `fetch_policy` then require one.

    Request throttling:
    Applies the rate limit set in the config (section `parallel`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        objective: The information to find, including any source or freshness requirements.
        search_queries: Keyword queries of three to six words each, which may use search operators; two or three
            work best. The service ranks sources against them and against `objective`. Defaults to `objective`
            as the only query.
        mode: How much work the service puts into the search, in increasing order of latency and cost. `advanced`
            and `basic` cost five times `fast` and `turbo`. See
            <https://docs.parallel.ai/search/modes>.
        max_results: How many sources to return. Defaults to the service's setting of 10, and is capped at 20.
        source_policy: Which sources to search, as `include_domains`, `exclude_domains` and `after_date`. See
            <https://docs.parallel.ai/search/advanced-search-settings>.
        fetch_policy: How fresh the page contents have to be, as `max_age_seconds`, `timeout_seconds` and
            `disable_cache_fallback`. Defaults to the service's setting, which serves its cached index.
        session_id: Application-owned conversation ID, a UUID or a hex string of at least 32 characters. Pass the
            same ID for related search and extract calls; omit it for standalone requests. Use a new opaque ID for
            a new conversation.

    Returns:
        A dictionary with the following entries:

        - `search_id` (`str`): Parallel's ID for this search, for support requests
        - `results` (`list`): the sources, ordered by decreasing relevance, each with

            - `url` (`str`): the source URL
            - `title` (`str | None`): the page title
            - `publish_date` (`str | None`): the publication date, as `YYYY-MM-DD`
            - `excerpts` (`list[str]`): passages relevant to `objective`

        - `warnings` (`list | None`): the adjustments the service made to the request, each with a `type`,
            a `message` and an optional `detail`
        - `usage` (`list | None`): what the request billed, each entry with a `name` and a `count`
        - `session_id` (`str`): the conversation ID, echoed back, or one generated by the service

    Examples:
        Search for each row's question:

        >>> from pixeltable.functions import parallel
        >>> tbl.add_computed_column(sources=parallel.search(tbl.question))

        Read the first source's excerpts out of the response:

        >>> tbl.select(tbl.sources.results[0].excerpts).collect()

        Search with keyword queries of their own, in the highest-quality mode:

        >>> tbl.add_computed_column(
        ...     sources=parallel.search(
        ...         tbl.question, search_queries=tbl.keyword_queries, mode='advanced'
        ...     )
        ... )

        Make search available to an LLM using the existing tool-calling API:

        >>> tools = pxt.tools(parallel.search)
    """
    advanced_settings = _only_set(max_results=max_results, source_policy=source_policy, fetch_policy=fetch_policy)
    request = _only_set(
        objective=objective,
        search_queries=[objective] if search_queries is None else search_queries,
        mode=mode,
        session_id=session_id,
        advanced_settings=advanced_settings or None,
    )
    return cast(ParallelSearchResponse, await _client().search(request))


class ParallelExtractResult(TypedDict):
    url: str
    title: NotRequired[str | None]
    publish_date: NotRequired[str | None]
    acquired_at: NotRequired[str | None]
    excerpts: list[str]
    full_content: NotRequired[str | None]
    links: NotRequired[list[str] | None]


class ParallelExtractError(TypedDict):
    url: str
    error_type: str
    http_status_code: int | None
    content: str | None


class ParallelExtractResponse(TypedDict):
    extract_id: str
    results: list[ParallelExtractResult]
    errors: list[ParallelExtractError]
    warnings: list[ParallelWarning] | None
    usage: NotRequired[list[ParallelUsage] | None]
    session_id: str


@pxt.udf(is_deterministic=False, resource_pool='request-rate:parallel')
async def extract(
    url: str,
    *,
    objective: str | None = None,
    search_queries: list[str] | None = None,
    full_content: bool | None = None,
    fetch_policy: ParallelFetchPolicy | None = None,
    links: bool | None = None,
    allow_live_fetch: bool | None = None,
    session_id: str | None = None,
) -> ParallelExtractResponse:
    """Extract relevant excerpts, or the whole text, from a web page.

    Equivalent to the Parallel Extract API endpoint.
    For additional details, see: <https://docs.parallel.ai/extract/extract-quickstart>

    Without an API key, the free endpoint returns excerpts or full content; `fetch_policy` and `links` then
    require one, and `allow_live_fetch` is the freshness setting it offers in place of `fetch_policy`.

    Request throttling:
    Applies the rate limit set in the config (section `parallel`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        url: An HTTP or HTTPS page URL.
        objective: Information to extract, up to 200 characters. Omit for general page excerpts.
        search_queries: Keyword queries of three to six words each, which focus the excerpts alongside `objective`;
            pass the queries of the search that produced `url`.
        full_content: Return the whole page as markdown instead of excerpts. A long page can run to tens of
            thousands of tokens. Defaults to the service's setting, which returns excerpts.
        fetch_policy: How fresh the page contents have to be, as `max_age_seconds`, `timeout_seconds` and
            `disable_cache_fallback`. Defaults to the service's setting, which decides from `url` and `objective`.
            Requires an API key.
        links: Also return the page's outgoing links. Requires an API key.
        allow_live_fetch: Whether the free endpoint may read the page live rather than serve its cached copy.
            Defaults to the service's setting of `True`. With an API key, use `fetch_policy` instead.
        session_id: Application-owned conversation ID; pass the ID used by the related searches.

    Returns:
        A dictionary with the following entries:

        - `extract_id` (`str`): Parallel's ID for this extraction, for support requests
        - `results` (`list`): one entry if the page was read, otherwise empty, with

            - `url` (`str`): the page URL
            - `title` (`str | None`): the page title
            - `publish_date` (`str | None`): the publication date, as `YYYY-MM-DD`
            - `excerpts` (`list[str]`): passages relevant to `objective`
            - `full_content` (`str | None`): the whole page as markdown, if `full_content` asked for it
            - `links` (`list[str] | None`): the page's outgoing links, if `links` asked for them

        - `errors` (`list`): one entry if the page could not be read, otherwise empty, with

            - `url` (`str`): the page URL
            - `error_type` (`str`): Parallel's error classification, e.g. `fetch_error`
            - `http_status_code` (`int | None`): the status returned by the page
            - `content` (`str | None`): the error body, if the page returned one

        - `warnings` (`list | None`): the adjustments the service made to the request, each with a `type`,
            a `message` and an optional `detail`
        - `usage` (`list | None`): what the request billed, each entry with a `name` and a `count`
        - `session_id` (`str`): the conversation ID, echoed back, or one generated by the service

        A page that could not be read is reported in `errors` rather than as a UDF error.

    Examples:
        Extract excerpts for a column of page URLs:

        >>> from pixeltable.functions import parallel
        >>> tbl.add_computed_column(page=parallel.extract(tbl.url))

        Read the excerpts, and the reason a page could not be read, out of the response:

        >>> tbl.select(
        ...     tbl.page.results[0].excerpts, tbl.page.errors[0].error_type
        ... ).collect()

        Group a row's search and extraction under the conversation ID stored alongside it:

        >>> tbl.add_computed_column(
        ...     page=parallel.extract(tbl.url, session_id=tbl.conversation)
        ... )
    """
    advanced_settings = _only_set(full_content=full_content, fetch_policy=fetch_policy, links=links)
    request = _only_set(
        urls=[url],
        objective=objective,
        search_queries=search_queries,
        allow_live_fetch=allow_live_fetch,
        session_id=session_id,
        advanced_settings=advanced_settings or None,
    )
    return cast(ParallelExtractResponse, await _client().extract(request))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
