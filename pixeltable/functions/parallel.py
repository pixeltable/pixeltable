"""
Web search and page extraction through [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).

In order to use these UDFs, you must first `pip install mcp`. Search uses Fast mode without an account or API key,
subject to the service's anonymous usage limits. Set `PARALLEL_API_KEY`, or `api_key` in the `parallel` configuration
section, to use an existing Parallel account. A configured key is used as supplied; authentication errors do not fall
back to anonymous requests.

These UDFs run only when called. Include them explicitly in `pxt.tools()` to let a model use them. Pass a
`session_id` to group the calls a conversation makes; Pixeltable does not infer conversation boundaries from rows
or keep a shared conversation ID.
"""

import json
from typing import Any, TypedDict, cast
from urllib.parse import urlsplit

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import Env, register_client
from pixeltable.func.mcp import mcp_session
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names

_MCP_URL = 'https://search.parallel.ai/mcp'
_REQUEST_TIMEOUT = 60.0


class _ParallelClient:
    def __init__(self, api_key: str | None):
        self.headers = {'User-Agent': f'Pixeltable/{pxt.__version__} (+https://github.com/pixeltable/pixeltable)'}
        if api_key is not None:
            self.headers['Authorization'] = f'Bearer {api_key}'

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        Env.get().require_package('mcp', [2])
        import mcp

        # Each request owns its transport; the cached client holds no mutable conversation state or open sockets.
        try:
            async with mcp_session(_MCP_URL, headers=self.headers, timeout=_REQUEST_TIMEOUT) as session:
                result = await session.call_tool(tool, arguments)
        except ExceptionGroup as group:
            # AnyIO wraps transport errors; expose the underlying diagnostic to computed-column error reporting.
            error: BaseException = group
            while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
                error = error.exceptions[0]
            raise error from None

        if not isinstance(result, mcp.types.CallToolResult):
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR, 'Parallel returned an unsupported MCP result.'
            )
        text = '\n'.join(item.text for item in result.content if isinstance(item, mcp.types.TextContent))
        if result.is_error:
            raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, f'Parallel {tool} failed: {text}')
        if isinstance(result.structured_content, dict):
            return result.structured_content
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR, 'Parallel returned an invalid JSON result.')
        return parsed


@register_client('parallel', credential_param='api_key')
def _(api_key: str | None = None) -> _ParallelClient:
    return _ParallelClient(api_key)


def _session_args(session_id: str | None) -> dict[str, str]:
    if session_id is None:
        return {}
    if not 1 <= len(session_id) <= 100:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'session_id must contain 1 to 100 characters.')
    return {'session_id': session_id}


class ParallelPageResult(TypedDict):
    """One source page, as both `search` and `fetch` report it."""

    url: str
    title: str | None
    publish_date: str | None
    excerpts: list[str] | None


class ParallelSearchResponse(TypedDict):
    search_id: str
    results: list[ParallelPageResult]
    warnings: list[dict] | None
    session_id: str


@pxt.udf(is_deterministic=False, resource_pool='request-rate:parallel')
async def search(query: str, *, session_id: str | None = None) -> ParallelSearchResponse:
    """Search the web and return source URLs, titles and relevant excerpts.

    Uses Parallel Search MCP in Fast mode by default, without an account or API key.

    Request throttling:
    Applies the rate limit set in the config (section `parallel`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        query: The information to find, including any source or freshness requirements.
        session_id: Application-owned conversation ID. Pass the same ID for related search and fetch calls;
            omit it for standalone requests. Use a new opaque ID for a new conversation.

    Returns:
        A dictionary with the following entries:

        - `search_id` (`str`): Parallel's ID for this search, for support requests
        - `results` (`list`): the sources, ordered by decreasing relevance, each with

            - `url` (`str`): the source URL
            - `title` (`str | None`): the page title
            - `publish_date` (`str | None`): the publication date, as `YYYY-MM-DD`
            - `excerpts` (`list[str] | None`): passages relevant to `query`

        - `warnings` (`list[dict] | None`): non-fatal adjustments Parallel made to the request
        - `session_id` (`str`): the conversation ID, echoed back, or one generated by the service

    Examples:
        Search for each row's question:

        >>> from pixeltable.functions import parallel
        >>> tbl.add_computed_column(sources=parallel.search(tbl.question))

        Read the first source's excerpts out of the response:

        >>> tbl.select(tbl.sources.results[0].excerpts).collect()

        Make search available to an LLM using the existing tool-calling API:

        >>> tools = pxt.tools(parallel.search)
    """
    if not query.strip():
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'query must not be empty.')
    arguments = {'objective': query, 'search_queries': [query], **_session_args(session_id)}
    return cast(ParallelSearchResponse, await get_runtime().get_client('parallel').call('web_search', arguments))


class ParallelFetchFailure(TypedDict):
    url: str
    error_type: str
    http_status_code: int | None
    content: str | None


class ParallelFetchResponse(TypedDict):
    extract_id: str
    results: list[ParallelPageResult]
    errors: list[ParallelFetchFailure]
    warnings: list[dict] | None
    session_id: str


@pxt.udf(is_deterministic=False, resource_pool='request-rate:parallel')
async def fetch(url: str, *, objective: str | None = None, session_id: str | None = None) -> ParallelFetchResponse:
    """Fetch relevant excerpts from a web page with Parallel Search MCP.

    Request throttling:
    Applies the rate limit set in the config (section `parallel`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        url: An HTTP or HTTPS page URL.
        objective: Information to extract, up to 200 characters. Omit for general page excerpts.
        session_id: Application-owned conversation ID; pass the ID used by the related searches.

    Returns:
        A dictionary with the following entries:

        - `extract_id` (`str`): Parallel's ID for this extraction, for support requests
        - `results` (`list`): one entry if the page was fetched, otherwise empty, with

            - `url` (`str`): the page URL
            - `title` (`str | None`): the page title
            - `publish_date` (`str | None`): the publication date, as `YYYY-MM-DD`
            - `excerpts` (`list[str] | None`): passages relevant to `objective`

        - `errors` (`list`): one entry if the page could not be fetched, otherwise empty, with

            - `url` (`str`): the page URL
            - `error_type` (`str`): Parallel's error classification, e.g. `http_error`
            - `http_status_code` (`int | None`): the status returned by the page
            - `content` (`str | None`): the error body, if the page returned one

        - `warnings` (`list[dict] | None`): non-fatal adjustments Parallel made to the request
        - `session_id` (`str`): the conversation ID, echoed back, or one generated by the service

        A page that could not be fetched is reported in `errors` rather than as a UDF error.

    Examples:
        Fetch excerpts for a column of page URLs:

        >>> from pixeltable.functions import parallel
        >>> tbl.add_computed_column(page=parallel.fetch(tbl.url))

        Read the excerpts, and the reason a page could not be fetched, out of the response:

        >>> tbl.select(
        ...     tbl.page.results[0].excerpts, tbl.page.errors[0].error_type
        ... ).collect()

        Group a row's search and fetch under the conversation ID stored alongside it:

        >>> tbl.add_computed_column(
        ...     page=parallel.fetch(tbl.url, session_id=tbl.conversation)
        ... )
    """
    try:
        parsed = urlsplit(url)
        valid_url = parsed.scheme in ('http', 'https') and parsed.hostname is not None
    except ValueError:
        valid_url = False
    if not valid_url:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'url must be an HTTP or HTTPS URL.')
    if objective is not None and len(objective) > 200:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'objective must contain at most 200 characters.')
    arguments: dict[str, Any] = {'urls': [url], **_session_args(session_id)}
    if objective is not None:
        arguments['objective'] = objective
    return cast(ParallelFetchResponse, await get_runtime().get_client('parallel').call('web_fetch', arguments))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
