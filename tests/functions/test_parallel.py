import contextlib
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.functions import parallel
from pixeltable.runtime import reset_runtime

from ..utils import pxt_raises, rerun_on_network_error, skip_test_if_not_installed, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

_OBJECTIVE = 'What is the Pixeltable Python library for?'
_QUERIES = ['Pixeltable python library', 'Pixeltable multimodal data']
_URL = 'https://pixeltable.com/'
_MISSING_URL = 'https://pixeltable.com/no-such-page-6f3a1c'

_SEARCH_RESPONSE = {
    'search_id': 'search_fixture',
    'results': [{'url': _URL, 'title': 'Pixeltable', 'publish_date': '2026-01-15', 'excerpts': ['An excerpt.']}],
    'warnings': None,
    'usage': [{'name': 'sku_search', 'count': 1}],
    'session_id': 'session_fixture',
}


@contextlib.contextmanager
def _configure(tmp_path: Path, api_key: str | None) -> Iterator[pytest.MonkeyPatch]:
    """Reconfigure the `parallel` client, from a config file of the test's own so the developer's supplies nothing."""
    project_root = Config.get().project_root
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('PIXELTABLE_CONFIG', str(tmp_path / 'config.toml'))
            if api_key is None:
                mp.delenv('PARALLEL_API_KEY', raising=False)
            else:
                mp.setenv('PARALLEL_API_KEY', api_key)
            Config.init(reinit=True, project_root=project_root)
            reset_runtime()
            yield mp
    finally:
        reset_runtime()
        Config.init(reinit=True, project_root=project_root)


@pytest.fixture
def anonymous(uses_db: None, tmp_path: Path) -> Iterator[None]:
    """Leave the client without an api_key, so that the free endpoint serves the test."""
    skip_test_if_not_installed('mcp')  # the free endpoint speaks MCP, and the client needs the SDK to reach it
    with _configure(tmp_path, None):
        yield


@pytest.fixture
def authenticated(uses_db: None) -> None:
    if not os.environ.get('PARALLEL_API_KEY'):
        pytest.skip('`PARALLEL_API_KEY` is not set.')


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI response calling one tool, in place of a model's, so that the tool tests need no LLM."""
    return {'choices': [{'message': {'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}


@pytest.mark.expensive
@pytest.mark.remote_api
def test_free_tier(anonymous: None) -> None:
    from pixeltable.functions.openai import invoke_tools

    tools = pxt.tools(parallel.search, parallel.extract)
    t = pxt.create_table(
        'test_tbl',
        {
            'id': pxt.Int,
            'objective': pxt.String | None,
            'url': pxt.String | None,
            'missing_url': pxt.String | None,
            'response': pxt.Json,
        },
    )
    t.add_computed_column(sources=parallel.search(t.objective, search_queries=_QUERIES))
    t.add_computed_column(
        page=parallel.extract(t.url, objective='What is Pixeltable?', full_content=True, allow_live_fetch=False)
    )
    t.add_computed_column(missing_page=parallel.extract(t.missing_url))
    t.add_computed_column(tool_calls=invoke_tools(tools, t.response))

    rows = [
        {
            'id': 0,
            'objective': _OBJECTIVE,
            'url': _URL,
            'missing_url': _MISSING_URL,
            'response': _tool_call('search', {'objective': _OBJECTIVE, 'mode': 'fast'}),
        },
        # null inputs skip the direct calls, so this row makes only the extract tool call
        {'id': 1, 'response': _tool_call('extract', {'url': _URL})},
    ]
    validate_update_status(t.insert(rows), 2)
    first, second = t.order_by(t.id).collect()

    sources = first['sources']
    assert sources['search_id'].startswith('search_')
    assert len(sources['session_id']) > 0
    assert len(sources['results']) > 0
    assert all(result['url'].startswith('http') for result in sources['results'])
    assert all(len(result['excerpts']) > 0 for result in sources['results'])

    page = first['page']
    assert page['extract_id'].startswith('extract_')
    assert page['errors'] == []
    assert 'pixeltable' in page['results'][0]['full_content'].lower()

    missing_page = first['missing_page']
    assert missing_page['results'] == []
    assert missing_page['errors'][0]['url'] == _MISSING_URL
    assert missing_page['errors'][0]['http_status_code'] == 404

    assert first['tool_calls']['extract'] is None
    assert len(first['tool_calls']['search'][0]['results']) > 0
    assert second['tool_calls']['search'] is None
    assert len(second['tool_calls']['extract'][0]['results']) == 1


@pytest.mark.expensive
@pytest.mark.remote_api
@rerun_on_network_error()
class TestParallel:
    def test_search_with_key(self, authenticated: None) -> None:
        t = pxt.create_table('test_tbl', {'objective': pxt.String | None})
        t.add_computed_column(
            sources=parallel.search(t.objective, search_queries=_QUERIES, mode='basic', max_results=3)
        )

        validate_update_status(t.insert(objective=_OBJECTIVE), 1)
        response = t.select(t.sources).collect()['sources'][0]
        assert response['search_id'].startswith('search_')
        assert 0 < len(response['results']) <= 3
        assert all(len(result['excerpts']) > 0 for result in response['results'])
        assert all(item['count'] > 0 for item in response['usage'])

    def test_extract_with_key(self, authenticated: None) -> None:
        t = pxt.create_table('test_tbl', {'url': pxt.String | None})
        t.add_computed_column(page=parallel.extract(t.url, objective='What is Pixeltable?', links=True))

        validate_update_status(t.insert(url=_URL), 1)
        response = t.select(t.page).collect()['page'][0]
        assert response['errors'] == []
        assert len(response['results'][0]['excerpts']) > 0
        assert all(link.startswith('http') for link in response['results'][0]['links'])

    def test_tool_invocations_with_key(self, authenticated: None) -> None:
        from pixeltable.functions.openai import invoke_tools

        tools = pxt.tools(parallel.search)
        t = pxt.create_table('test_tbl', {'response': pxt.Json | None})
        t.add_computed_column(tool_calls=invoke_tools(tools, t.response))

        call = _tool_call('search', {'objective': _OBJECTIVE, 'mode': 'advanced', 'max_results': 2})
        validate_update_status(t.insert(response=call), 1)
        tool_calls = t.select(t.tool_calls).collect()['tool_calls'][0]
        assert 0 < len(tool_calls['search'][0]['results']) <= 2

    def test_rejected_key(self, uses_db: None, tmp_path: Path) -> None:
        with _configure(tmp_path, 'not-a-key'):
            t = pxt.create_table('test_tbl', {'objective': pxt.String | None})
            t.add_computed_column(sources=parallel.search(t.objective))
            with pxt_raises(pxt.ErrorCode.PROVIDER_AUTH_ERROR, match='PARALLEL_API_KEY'):
                t.insert(objective=_OBJECTIVE)


@pytest.mark.parametrize(
    ('udf', 'option'),
    [
        (parallel.search.using(mode='advanced'), 'mode'),
        (parallel.search.using(max_results=5), 'max_results'),
        (parallel.search.using(source_policy={'include_domains': ['pixeltable.com']}), 'source_policy'),
        (parallel.search.using(fetch_policy={'max_age_seconds': 600}), 'fetch_policy'),
    ],
)
def test_search_option_needs_key(anonymous: None, udf: pxt.Function, option: str) -> None:
    t = pxt.create_table('test_tbl', {'objective': pxt.String | None})
    t.add_computed_column(sources=udf(t.objective))
    with pxt_raises(pxt.ErrorCode.MISSING_CREDENTIALS, match=option):
        t.insert(objective=_OBJECTIVE)


@pytest.mark.parametrize(
    ('udf', 'option'),
    [
        (parallel.extract.using(fetch_policy={'max_age_seconds': 600}), 'fetch_policy'),
        (parallel.extract.using(links=True), 'links'),
    ],
)
def test_extract_option_needs_key(anonymous: None, udf: pxt.Function, option: str) -> None:
    t = pxt.create_table('test_tbl', {'url': pxt.String | None})
    t.add_computed_column(page=udf(t.url))
    with pxt_raises(pxt.ErrorCode.MISSING_CREDENTIALS, match=option):
        t.insert(url=_URL)


def test_extract_live_fetch_needs_no_key(uses_db: None, tmp_path: Path) -> None:
    # the mirror case: the free endpoint's freshness setting is not one the REST API accepts
    with _configure(tmp_path, 'test-key'):
        t = pxt.create_table('test_tbl', {'url': pxt.String | None})
        t.add_computed_column(page=parallel.extract(t.url, allow_live_fetch=False))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='allow_live_fetch'):
            t.insert(url=_URL)


@contextlib.contextmanager
def _serving(handler: type[BaseHTTPRequestHandler]) -> Iterator[int]:
    """`handler` on a port of its own, for the duration of the context."""
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def rate_limited_service(uses_db: None, tmp_path: Path) -> Iterator[list[int]]:
    """A stand-in for the REST API that rejects the first request with a 429, which the service can't do on demand.

    Yields the statuses it served, in order.
    """
    statuses: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers['Content-Length']))
            status = 429 if len(statuses) == 0 else 200
            statuses.append(status)
            body = json.dumps({'error': 'rate limited'} if status == 429 else _SEARCH_RESPONSE).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with _serving(Handler) as port, _configure(tmp_path, 'test-key') as mp:
        mp.setattr(parallel, '_REST_BASE_URL', f'http://127.0.0.1:{port}')
        yield statuses


def test_rate_limit_is_retried(rate_limited_service: list[int]) -> None:
    t = pxt.create_table('test_tbl', {'objective': pxt.String | None})
    t.add_computed_column(sources=parallel.search(t.objective))

    validate_update_status(t.insert(objective=_OBJECTIVE), 1)
    assert rate_limited_service == [429, 200]
    assert t.select(t.sources).collect()['sources'][0]['search_id'] == _SEARCH_RESPONSE['search_id']


class _McpEndpoint:
    """Enough of MCP's Streamable HTTP transport to stand in for the free endpoint.

    It fails the first tool call with `first_status`, a status the service won't produce on demand, and serves
    every later one. `tool_call_statuses` and `session_ids` record what it did, so that a test can tell a retry on
    the same session from one on a session opened after the first was dropped.
    """

    first_status: int
    tool_call_statuses: list[int]
    session_ids: list[str]

    def __init__(self, first_status: int):
        self.first_status = first_status
        self.tool_call_statuses = []
        self.session_ids = []

    def handler(self) -> type[BaseHTTPRequestHandler]:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_GET(self) -> None:
                self._reply(405)  # no server-initiated event stream

            def do_DELETE(self) -> None:
                self._reply(405)  # no session termination

            def do_POST(self) -> None:
                message = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if 'id' not in message:
                    self._reply(202)  # a notification, which takes no response
                elif message['method'] == 'initialize':
                    endpoint.session_ids.append(uuid.uuid4().hex)
                    result = {
                        'protocolVersion': message['params']['protocolVersion'],
                        'capabilities': {'tools': {}},
                        'serverInfo': {'name': 'parallel-stand-in', 'version': '0'},
                    }
                    self._reply(200, message['id'], result, session_id=endpoint.session_ids[-1])
                elif message['method'] == 'tools/list':
                    # no output schema, so that the SDK takes the result's text content as it comes
                    tools = [
                        {'name': name, 'description': name, 'inputSchema': {'type': 'object'}}
                        for name in ('web_search', 'web_fetch')
                    ]
                    self._reply(200, message['id'], {'tools': tools})
                else:
                    assert message['method'] == 'tools/call', message['method']
                    status = endpoint.first_status if len(endpoint.tool_call_statuses) == 0 else 200
                    endpoint.tool_call_statuses.append(status)
                    if status != 200:
                        self._reply(status)
                    else:
                        content = [{'type': 'text', 'text': json.dumps(_SEARCH_RESPONSE)}]
                        self._reply(200, message['id'], {'content': content, 'isError': False})

            def _reply(
                self, status: int, request_id: Any = None, result: Any = None, session_id: str | None = None
            ) -> None:
                body = b''
                if result is not None:
                    body = json.dumps({'jsonrpc': '2.0', 'id': request_id, 'result': result}).encode()
                self.send_response(status)
                if len(body) > 0:
                    self.send_header('Content-Type', 'application/json')
                if session_id is not None:
                    self.send_header('Mcp-Session-Id', session_id)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler


@pytest.fixture
def failing_mcp_endpoint(request: pytest.FixtureRequest, uses_db: None, tmp_path: Path) -> Iterator[_McpEndpoint]:
    """The free endpoint, stood in for by a server that fails the first tool call with the parametrized status."""
    skip_test_if_not_installed('mcp')
    endpoint = _McpEndpoint(request.param)
    with _serving(endpoint.handler()) as port, _configure(tmp_path, None) as mp:
        mp.setattr(parallel, '_MCP_URL', f'http://127.0.0.1:{port}/mcp')
        yield endpoint


@pytest.mark.parametrize('failing_mcp_endpoint', [429], indirect=True)
def test_free_rate_limit_is_retried(failing_mcp_endpoint: _McpEndpoint) -> None:
    # a 429 reaches the scheduler as RATE_LIMITED, which retries the row on the session it already has
    t = pxt.create_table('test_tbl', {'objective': pxt.String | None})
    t.add_computed_column(sources=parallel.search(t.objective))

    validate_update_status(t.insert(objective=_OBJECTIVE), 1)
    assert failing_mcp_endpoint.tool_call_statuses == [429, 200]
    assert len(failing_mcp_endpoint.session_ids) == 1
    assert t.select(t.sources).collect()['sources'][0]['search_id'] == _SEARCH_RESPONSE['search_id']


@pytest.mark.parametrize('failing_mcp_endpoint', [404], indirect=True)
def test_expired_session_is_reopened(failing_mcp_endpoint: _McpEndpoint) -> None:
    # a 404 means the server has forgotten the session, so the call reconnects and runs once more
    t = pxt.create_table('test_tbl', {'objective': pxt.String | None})
    t.add_computed_column(sources=parallel.search(t.objective))

    validate_update_status(t.insert(objective=_OBJECTIVE), 1)
    assert failing_mcp_endpoint.tool_call_statuses == [404, 200]
    assert len(failing_mcp_endpoint.session_ids) == 2
    assert t.select(t.sources).collect()['sources'][0]['search_id'] == _SEARCH_RESPONSE['search_id']
