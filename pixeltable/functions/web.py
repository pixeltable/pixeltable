"""
Web search and page extraction through [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).

Install with `pip install 'pixeltable[web]'`. Search uses Fast mode without an account or API key, subject to
the service's anonymous usage limits. Set `PARALLEL_API_KEY`, or `api_key` in the `parallel` configuration section,
to use an existing Parallel account. A configured key is used as supplied; authentication errors do not fall back
to anonymous requests.

These UDFs run only when called. Include them explicitly in `pxt.tools()` to let a model use them. Applications
that group calls into a conversation should bind their own stable, opaque `session_id` with `using()` and reuse
it for related searches and fetches. Start a new ID for a new conversation. Pixeltable does not infer conversation
boundaries from rows or keep a shared conversation ID.
"""

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names

_MCP_URL = 'https://search.parallel.ai/mcp'
_REQUEST_TIMEOUT = 60.0


class _ParallelClient:
    def __init__(self, api_key: str | None):
        self.headers = {'User-Agent': f'Pixeltable/{pxt.__version__} (+https://github.com/pixeltable/pixeltable)'}
        if api_key is not None:
            self.headers['Authorization'] = f'Bearer {api_key}'

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict:
        Env.get().require_package('mcp', [2])
        import httpx2
        import mcp
        from mcp.client.streamable_http import streamable_http_client

        # Each request owns its transport; the cached client holds no mutable conversation state or open sockets.
        try:
            async with asyncio.timeout(_REQUEST_TIMEOUT):
                async with (
                    httpx2.AsyncClient(headers=self.headers, timeout=_REQUEST_TIMEOUT) as http_client,
                    streamable_http_client(_MCP_URL, http_client=http_client) as (read_stream, write_stream),
                    mcp.ClientSession(read_stream, write_stream, read_timeout_seconds=_REQUEST_TIMEOUT) as session,
                ):
                    await session.initialize()
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


@pxt.udf(is_deterministic=False)
async def search(query: str, *, session_id: str | None = None) -> dict:
    """Search the web and return source URLs, titles and relevant excerpts.

    Uses Parallel Search MCP in Fast mode by default, without an account or API key.

    Args:
        query: The information to find, including any source or freshness requirements.
        session_id: Application-owned conversation ID. Bind the same ID for related search and fetch calls;
            omit it for standalone requests. Use a new opaque ID for a new conversation.

    Returns:
        The search response, including a `results` list with source URLs and excerpts.

    Examples:
        Search for each row's question:

        >>> from pixeltable.functions import web
        >>> tbl.add_computed_column(sources=web.search(tbl.question))

        Make search available to an LLM using the existing tool-calling API:

        >>> tools = pxt.tools(web.search)
    """
    if not query.strip():
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'query must not be empty.')
    arguments = {'objective': query, 'search_queries': [query], **_session_args(session_id)}
    return await get_runtime().get_client('parallel').call('web_search', arguments)


@pxt.udf(is_deterministic=False)
async def fetch(url: str, *, objective: str | None = None, session_id: str | None = None) -> dict:
    """Fetch relevant excerpts from a web page with Parallel Search MCP.

    Args:
        url: An HTTP or HTTPS page URL.
        objective: Information to extract, up to 200 characters. Omit for general page excerpts.
        session_id: Application-owned conversation ID; reuse the ID from related searches.

    Returns:
        The extraction response, including `results` with page URLs and excerpts and `errors` for pages that
        could not be fetched. A per-page extraction error remains part of the response.

    Examples:
        Fetch excerpts for a column of page URLs:

        >>> from pixeltable.functions import web
        >>> tbl.add_computed_column(page=web.fetch(tbl.url))

        Bind the application's conversation ID before exposing tools to a model:

        >>> tools = pxt.tools(
        ...     pxt.tool(web.search.using(session_id=conversation_id), name='search'),
        ...     pxt.tool(web.fetch.using(session_id=conversation_id), name='fetch'),
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
    return await get_runtime().get_client('parallel').call('web_fetch', arguments)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
