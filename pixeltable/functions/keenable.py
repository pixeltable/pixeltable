"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) that wrap
[Keenable](https://keenable.ai/)'s web search and page fetch APIs.

No account is needed: without an API key, the UDFs call Keenable's free public endpoints, which allow 1,000 requests
per hour per IP address. Specify `KEENABLE_API_KEY`, or `api_key` in the `keenable` section of the Pixeltable config
file, to call the keyed endpoints, which have higher limits.
"""

import logging
from typing import Any, NotRequired, TypedDict, cast

import aiohttp

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import register_client
from pixeltable.runtime import get_runtime
from pixeltable.utils.http import extract_retry_delay_from_headers

_logger = logging.getLogger(__name__)

_BASE_URL = 'https://api.keenable.ai'
_REQUEST_TIMEOUT = 60.0
_USER_AGENT = f'Pixeltable/{pxt.__version__} (+https://github.com/pixeltable/pixeltable)'
# the free endpoints ask callers without a key to name their application
_APP_TITLE = 'Pixeltable'

# How a failed response maps onto Pixeltable's error codes. The code decides whether the scheduler retries the row:
# it retries RATE_LIMITED, PROVIDER_OVERLOADED and an uncaused PROVIDER_ERROR, and nothing else.
_STATUS_ERROR_CODES: dict[int, excs.ErrorCode] = {
    400: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    401: excs.ErrorCode.PROVIDER_AUTH_ERROR,
    402: excs.ErrorCode.PROVIDER_AUTH_ERROR,
    403: excs.ErrorCode.PROVIDER_AUTH_ERROR,
    404: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    405: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    408: excs.ErrorCode.PROVIDER_TIMEOUT,
    422: excs.ErrorCode.PROVIDER_BAD_REQUEST,
    429: excs.ErrorCode.RATE_LIMITED,
    503: excs.ErrorCode.PROVIDER_OVERLOADED,
    504: excs.ErrorCode.PROVIDER_TIMEOUT,
}


def _status_error(status: int, body: str = '', retry_after: float | None = None) -> excs.ExternalServiceError:
    message = f'Keenable returned HTTP {status}.'
    if len(body) > 0:
        message += f' {body[:500]}'
    if status in (401, 403):
        message += '\nCheck KEENABLE_API_KEY, or `api_key` in the `keenable` configuration section.'
    return excs.ExternalServiceError(
        _STATUS_ERROR_CODES.get(status, excs.ErrorCode.PROVIDER_ERROR),
        message,
        provider='keenable',
        status_code=status,
        retry_after=retry_after,
    )


def _only_set(**fields: Any) -> dict[str, Any]:
    """The fields a caller gave a value, so that the service applies its own defaults to the rest."""
    return {name: value for name, value in fields.items() if value is not None}


class _KeenableClient:
    """Calls Keenable's search and fetch endpoints: the keyed ones with an API key, the free public ones without."""

    _request_headers: dict[str, str]
    _path_suffix: str
    _session: aiohttp.ClientSession | None

    def __init__(self, api_key: str | None):
        self._request_headers = {'Accept': 'application/json', 'User-Agent': _USER_AGENT}
        if api_key is not None:
            self._request_headers['X-API-Key'] = api_key
            self._path_suffix = ''
        else:
            self._request_headers['X-Keenable-Title'] = _APP_TITLE
            self._path_suffix = '/public'
        self._session = None  # defer session creation until we have a running event loop

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(base_url=_BASE_URL)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        path = endpoint + self._path_suffix
        async with self._get_session().request(
            method, path, headers=self._request_headers, timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT), **kwargs
        ) as resp:
            if resp.status != 200:
                raise _status_error(resp.status, await resp.text(), extract_retry_delay_from_headers(resp.headers))
            result = await resp.json()
            if not isinstance(result, dict):
                raise excs.ExternalServiceError(
                    excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    f'Keenable {path} returned {type(result).__name__} rather than an object.',
                    provider='keenable',
                )
            return result

    async def search(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', '/v1/search', json=request)

    async def fetch(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', '/v1/fetch', json=request)


@register_client('keenable', credential_param='api_key')
def _(api_key: str | None = None) -> _KeenableClient:
    return _KeenableClient(api_key)


def _client() -> _KeenableClient:
    return get_runtime().get_client('keenable')


class KeenableSearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    published_at: NotRequired[str | None]
    acquired_at: NotRequired[str | None]


class KeenableSearchResponse(TypedDict):
    query: str
    mode: str
    results: list[KeenableSearchResult]
    warning: NotRequired[str | None]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:keenable')
async def search(
    query: str,
    *,
    max_results: int | None = None,
    site: str | None = None,
    exclude_domains: list[str] | None = None,
    published_after: str | None = None,
    published_before: str | None = None,
    snippet_max_length: int | None = None,
) -> KeenableSearchResponse:
    """Search the web and return source URLs, titles and query-relevant snippets.

    Equivalent to the Keenable Search API endpoint.
    For additional details, see: <https://docs.keenable.ai/api-reference>

    Request throttling:
    Applies the rate limit set in the config (section `keenable`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM. The free endpoint allows 1,000 requests per hour per IP address,
    so set a lower `rate_limit` for large tables without a key.

    Args:
        query: The search query.
        max_results: How many results to return. Defaults to the service's setting of 10.
        site: Restrict results to this domain.
        exclude_domains: Never return results from these hosts.
        published_after: Only return pages published after this date, as `YYYY-MM-DD`, ISO 8601, or a relative
            value such as `7d`.
        published_before: Only return pages published before this date.
        snippet_max_length: The maximum length of each snippet, in characters, from 180 to 10000.

    Returns:
        A dictionary with the following entries:

        - `query` (`str`): the query, echoed back
        - `mode` (`str`): the search mode the service used
        - `results` (`list`): the sources, ordered by decreasing relevance, each with

            - `url` (`str`): the source URL
            - `title` (`str`): the page title
            - `snippet` (`str`): a passage relevant to `query`
            - `published_at` (`str | None`): when the page was published, as ISO 8601
            - `acquired_at` (`str | None`): when the page was indexed, as ISO 8601

        - `warning` (`str | None`): present when the request was not answered as a plain search

    Examples:
        Search for each row's question:

        >>> from pixeltable.functions import keenable
        >>> tbl.add_computed_column(sources=keenable.search(tbl.question))

        Read the first source's snippet out of the response:

        >>> tbl.select(tbl.sources.results[0].snippet).collect()

        Make search available to an LLM using the existing tool-calling API:

        >>> tools = pxt.tools(keenable.search, keenable.fetch)
    """
    request = _only_set(
        query=query,
        max_results=max_results,
        site=site,
        exclude_domains=exclude_domains,
        published_after=published_after,
        published_before=published_before,
        snippet_max_length=snippet_max_length,
    )
    return cast(KeenableSearchResponse, await _client().search(request))


class KeenableFetchResponse(TypedDict):
    url: str
    title: str | None
    content: str
    truncated: bool
    total_chars: int
    author: NotRequired[str | None]
    published_at: NotRequired[str | None]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:keenable')
async def fetch(url: str, *, max_chars: int | None = None, live: bool | None = None) -> KeenableFetchResponse:
    """Fetch a web page and return its main content as markdown.

    Equivalent to the Keenable Fetch API endpoint.
    For additional details, see: <https://docs.keenable.ai/api-reference/fetch-post>

    Request throttling:
    Applies the rate limit set in the config (section `keenable`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        url: The URL of the page to fetch.
        max_chars: Cut the content to this many characters. `truncated` then says whether it was cut.
        live: Read the page from the site (`retrieval.max_age_hours: 0`), rather than from the service's index.

    Returns:
        A dictionary with the following entries:

        - `url` (`str`): the URL fetched, after redirects
        - `title` (`str | None`): the page title
        - `content` (`str`): the page's main content, as markdown
        - `truncated` (`bool`): whether `content` was cut to `max_chars`
        - `total_chars` (`int`): the length of the full content, before `max_chars` was applied
        - `author` (`str | None`): the author's name
        - `published_at` (`str | None`): when the page was published, as ISO 8601

        A page that does not exist is reported as a `PROVIDER_BAD_REQUEST` error on the row.

    Examples:
        Fetch the content of a column of page URLs:

        >>> from pixeltable.functions import keenable
        >>> tbl.add_computed_column(page=keenable.fetch(tbl.url, max_chars=20000))

        >>> tbl.select(tbl.page.content).collect()
    """
    request: dict[str, Any] = {'urls': [url]}
    content = _only_set(max_chars=max_chars)
    if content:
        request['content'] = content
    if live:
        request['retrieval'] = {'max_age_hours': 0}
    response = await _client().fetch(request)
    results = response.get('results') if isinstance(response, dict) else None
    page = results[0] if isinstance(results, list) and results else None
    if not isinstance(page, dict):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            'Keenable /v1/fetch returned no page in results[0].',
            provider='keenable',
        )
    if 'error' in page:
        err = page['error']
        if isinstance(err, dict):
            status = int(err.get('status') or 502)
            body = err.get('message') or err.get('error') or str(err)
        else:
            status, body = 502, str(err)
        raise _status_error(status, str(body))
    return cast(KeenableFetchResponse, page)


__all__ = ['fetch', 'search']


def __dir__() -> list[str]:
    return __all__
