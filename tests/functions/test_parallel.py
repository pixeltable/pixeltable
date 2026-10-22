import asyncio
import json
import pathlib
import uuid
from typing import TYPE_CHECKING, Any

import pytest

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.functions import openai, parallel

from ..utils import pxt_raises, reload_catalog

if TYPE_CHECKING:
    import httpx2
else:
    httpx2 = pytest.importorskip('httpx2')
pytest.importorskip('mcp')

pytestmark = pytest.mark.db_roots('local', reason='Mock HTTP transport runs in the local UDF process')

_SOURCE_URL = 'https://example.org/source'

# Responses in the shape the service documents, so that the tests exercise the declared TypedDicts.
_SEARCH_RESPONSE = {
    'search_id': 'search_fixture',
    'results': [
        {'url': _SOURCE_URL, 'title': 'A source', 'publish_date': '2026-01-15', 'excerpts': ['A source excerpt.']}
    ],
    'warnings': None,
    'session_id': 'session_fixture',
}

_FETCH_RESPONSE = {
    'extract_id': 'extract_fixture',
    'results': [
        {'url': _SOURCE_URL, 'title': 'A source', 'publish_date': '2026-01-15', 'excerpts': ['A source excerpt.']}
    ],
    'errors': [],
    'warnings': None,
    'session_id': 'session_fixture',
}


class _McpServer:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.calls: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.status = 200
        # how many requests `status` is returned for; None returns it for every request
        self.rejections: int | None = None
        self.block = False
        self.closed = 0

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200 and self.rejections != 0:
            if self.rejections is not None:
                self.rejections -= 1
            return httpx2.Response(self.status, json={'error': 'rejected'})
        if request.method != 'POST':
            return httpx2.Response(405)
        message = json.loads(request.content)
        if 'id' not in message:
            return httpx2.Response(202)
        if message['method'] == 'initialize':
            result = {
                'protocolVersion': message['params']['protocolVersion'],
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'search-fixture', 'version': '1'},
            }
        elif message['method'] == 'tools/list':
            result = {
                'tools': [
                    {'name': name, 'inputSchema': {'type': 'object', 'properties': {}}}
                    for name in ('web_search', 'web_fetch')
                ]
            }
        else:
            assert message['method'] == 'tools/call'
            self.calls.append(message['params'])
            if self.block:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.closed += 1
            payload = _SEARCH_RESPONSE if message['params']['name'] == 'web_search' else _FETCH_RESPONSE
            result = self.result or {'content': [{'type': 'text', 'text': json.dumps(payload)}]}
        return httpx2.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'result': result})


@pytest.fixture
def mcp_server(monkeypatch: pytest.MonkeyPatch, private_home: pathlib.Path) -> _McpServer:
    monkeypatch.delenv('PARALLEL_API_KEY', raising=False)
    server = _McpServer()
    original = httpx2.AsyncClient

    def client(**kwargs: Any) -> httpx2.AsyncClient:
        return original(**kwargs, transport=httpx2.MockTransport(server.handle))

    monkeypatch.setattr(httpx2, 'AsyncClient', client)
    return server


def test_search_and_fetch_columns(uses_db: None, mcp_server: _McpServer) -> None:
    t = pxt.create_table('parallel_test', {'query': pxt.String, 'conversation': pxt.String})
    t.add_computed_column(sources=parallel.search(t.query, session_id=t.conversation))
    t.add_computed_column(page=parallel.fetch(t.sources.results[0].url, session_id=t.conversation))
    # The declared TypedDicts give each response field a type, and confine JSON paths to the documented fields.
    assert t.sources.results[0].title.col_type.is_string_type()
    assert t.page.results[0].excerpts[0].col_type.is_string_type()
    with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='does not match the expected type schema'):
        _ = t.sources.result
    conversation = str(uuid.uuid4())
    t.insert(query='Pixeltable computed columns', conversation=conversation)
    row = t.collect()[0]
    assert row['sources'] == _SEARCH_RESPONSE
    assert row['page'] == _FETCH_RESPONSE
    assert [c['name'] for c in mcp_server.calls] == ['web_search', 'web_fetch']
    assert mcp_server.calls[0]['arguments'] == {
        'objective': 'Pixeltable computed columns',
        'search_queries': ['Pixeltable computed columns'],
        'session_id': conversation,
    }
    assert mcp_server.calls[1]['arguments'] == {'urls': [_SOURCE_URL], 'session_id': conversation}
    for request in mcp_server.requests:
        assert str(request.url) == 'https://search.parallel.ai/mcp'
        assert request.headers['user-agent'].startswith(f'Pixeltable/{pxt.__version__}')
        assert 'authorization' not in request.headers

    reload_catalog()
    t = pxt.get_table('parallel_test')
    t.insert(query='follow-up query', conversation=conversation)
    assert mcp_server.calls[-1]['arguments']['session_id'] == conversation
    assert t.count() == 2

    another = str(uuid.uuid4())
    t.insert(
        [
            {'query': 'independent conversation', 'conversation': another},
            {'query': 'original conversation follow-up', 'conversation': conversation},
        ]
    )
    search_calls = {c['arguments']['objective']: c['arguments'] for c in mcp_server.calls if c['name'] == 'web_search'}
    assert search_calls['independent conversation']['session_id'] == another
    assert search_calls['original conversation follow-up']['session_id'] == conversation


def test_tool_dispatch_and_choices(uses_db: None, mcp_server: _McpServer) -> None:
    # Scripted model responses exercise the production dispatcher without an inference API call.
    tools = pxt.tools(parallel.search, parallel.fetch)
    assert tools.choice(tool='fetch').tool == 'fetch'
    # a published tool exposes every parameter, session_id included, so the model can group its own calls
    assert 'session_id' in tools.ser_model()[0]['parameters']['properties']
    t = pxt.create_table('parallel_test', {'response': pxt.Json})
    t.add_computed_column(output=openai.invoke_tools(tools, t.response))
    restricted = pxt.tools(parallel.fetch)
    t.add_computed_column(restricted=openai.invoke_tools(restricted, t.response))
    t.insert(
        response={
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'search',
                                    'arguments': json.dumps({'query': 'current Pixeltable release'}),
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert t.collect()[0]['output'] == {'search': [_SEARCH_RESPONSE], 'fetch': None}
    assert len(mcp_server.calls) == 1
    assert 'session_id' not in mcp_server.calls[0]['arguments']

    t.insert(response={'choices': [{'message': {'content': 'No tool requested.'}}]})
    assert len(mcp_server.calls) == 1
    assert t.collect()[0]['restricted'] == {'fetch': None}


def test_tools_without_session_id(uses_db: None, mcp_server: _McpServer) -> None:
    # binding the parameter keeps the conversation ID with the application, as the agentic-workflows page shows
    tools = pxt.tools(pxt.tool(parallel.search.using(session_id=None), name='search'))
    assert 'session_id' not in tools.ser_model()[0]['parameters']['properties']
    t = pxt.create_table('parallel_test', {'response': pxt.Json})
    t.add_computed_column(output=openai.invoke_tools(tools, t.response))
    call = {'function': {'name': 'search', 'arguments': json.dumps({'query': 'current Pixeltable release'})}}
    t.insert(response={'choices': [{'message': {'tool_calls': [call]}}]})
    assert t.collect()[0]['output'] == {'search': [_SEARCH_RESPONSE]}
    assert 'session_id' not in mcp_server.calls[0]['arguments']


def test_bound_conversation_tools(uses_db: None, mcp_server: _McpServer) -> None:
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    t = pxt.create_table('parallel_test', {'response': pxt.Json})
    for name, conversation in [('first', first), ('second', second)]:
        tools = pxt.tools(
            pxt.tool(parallel.search.using(session_id=conversation), name='search'),
            pxt.tool(parallel.fetch.using(session_id=conversation), name='fetch'),
        )
        t.add_computed_column(**{name: openai.invoke_tools(tools, t.response)})
    for tool_name, args in [
        ('search', {'query': 'question'}),
        ('fetch', {'url': 'https://example.org/source'}),
        ('search', {'query': 'follow-up'}),
    ]:
        t.insert(
            response={
                'choices': [
                    {'message': {'tool_calls': [{'function': {'name': tool_name, 'arguments': json.dumps(args)}}]}}
                ]
            }
        )
    for conversation in (first, second):
        assert [c['name'] for c in mcp_server.calls if c['arguments'].get('session_id') == conversation] == [
            'web_search',
            'web_fetch',
            'web_search',
        ]
    assert all(row['first'] is not None and row['second'] is not None for row in t.collect())


def test_configured_key(uses_db: None, mcp_server: _McpServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PARALLEL_API_KEY', 'test-parallel-key')
    t = pxt.create_table('parallel_test', {'query': pxt.String})
    t.add_computed_column(sources=parallel.search(t.query))
    t.insert(query='test query')
    assert all(r.headers['authorization'] == 'Bearer test-parallel-key' for r in mcp_server.requests)


def test_key_from_config_section(uses_db: None, mcp_server: _McpServer, private_home: pathlib.Path) -> None:
    # the other path the module documents: `api_key` in the config file's `parallel` section, with no env var set
    private_home.mkdir(parents=True, exist_ok=True)
    (private_home / 'config.toml').write_text('[parallel]\napi_key = "test-config-key"\n', encoding='utf-8')
    Config.init(reinit=True, project_root=Config.get().project_root)
    t = pxt.create_table('parallel_test', {'query': pxt.String})
    t.add_computed_column(sources=parallel.search(t.query))
    t.insert(query='test query')
    assert mcp_server.requests
    assert all(r.headers['authorization'] == 'Bearer test-config-key' for r in mcp_server.requests)


@pytest.mark.parametrize('status', [401, 403])
def test_rejected_key_does_not_fall_back(
    uses_db: None, mcp_server: _McpServer, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv('PARALLEL_API_KEY', 'test-rejected-key')
    mcp_server.status = status
    t = pxt.create_table('parallel_test', {'query': pxt.String})
    t.add_computed_column(sources=parallel.search(t.query))
    # the transport drops the status and the body, so the rejection is reported from the status seen on the wire
    with pxt_raises(pxt.ErrorCode.PROVIDER_AUTH_ERROR, match=f'rejected the API key .HTTP {status}.*PARALLEL_API_KEY'):
        t.insert(query='test query')
    assert all(r.headers['authorization'] == 'Bearer test-rejected-key' for r in mcp_server.requests)
    assert not mcp_server.calls


@pytest.mark.parametrize(
    'status,error_code',
    [
        (402, pxt.ErrorCode.PROVIDER_ERROR),
        (408, pxt.ErrorCode.PROVIDER_TIMEOUT),
        (429, pxt.ErrorCode.RATE_LIMITED),
        (500, pxt.ErrorCode.PROVIDER_ERROR),
        (503, pxt.ErrorCode.PROVIDER_OVERLOADED),
        (504, pxt.ErrorCode.PROVIDER_TIMEOUT),
    ],
)
def test_status_errors(status: int, error_code: pxt.ErrorCode) -> None:
    error = parallel._status_error(status)
    assert error.error_code is error_code
    assert f'HTTP {status}' in error.message
    assert error.provider_http_status_code == status


def test_rate_limit_is_retried(uses_db: None, mcp_server: _McpServer) -> None:
    mcp_server.status = 429
    mcp_server.rejections = 1
    t = pxt.create_table('parallel_test', {'query': pxt.String})
    t.add_computed_column(sources=parallel.search(t.query))
    t.insert(query='test query')
    # the scheduler reads the status out of the message, retries the row, and the second attempt gets the result
    assert mcp_server.rejections == 0
    assert t.collect()[0]['sources'] == _SEARCH_RESPONSE


@pytest.mark.parametrize(
    'result,message',
    [
        ({'content': [{'type': 'text', 'text': 'search unavailable'}], 'isError': True}, 'search unavailable'),
        ({'content': [{'type': 'text', 'text': 'not JSON'}]}, 'invalid JSON result'),
    ],
)
def test_tool_errors(uses_db: None, mcp_server: _McpServer, result: dict, message: str) -> None:
    mcp_server.result = result
    t = pxt.create_table('parallel_test', {'query': pxt.String})
    t.add_computed_column(sources=parallel.search(t.query))
    assert t.insert(query='test query', on_error='ignore').num_excs == 1
    assert message in t.select(message=t.sources.errormsg).collect()[0]['message']


def test_fetch_partial_error(uses_db: None, mcp_server: _McpServer) -> None:
    payload = {
        'extract_id': 'extract_fixture',
        'results': [],
        'errors': [
            {'url': 'https://example.org/missing', 'error_type': 'http_error', 'http_status_code': 404, 'content': None}
        ],
        'warnings': None,
        'session_id': 'session_fixture',
    }
    mcp_server.result = {'content': [], 'structuredContent': payload}
    t = pxt.create_table('parallel_test', {'url': pxt.String})
    t.add_computed_column(page=parallel.fetch(t.url, objective='Find the title'))
    t.insert(url='https://example.org/missing')
    assert t.collect()[0]['page'] == payload
    # The declared TypedDicts type the fields of a per-page error, which is reported in the response rather than raised.
    error = t.select(error_type=t.page.errors[0].error_type, status=t.page.errors[0].http_status_code).collect()[0]
    assert error == {'error_type': 'http_error', 'status': 404}
    assert mcp_server.calls[0]['arguments']['objective'] == 'Find the title'


def test_invalid_arguments(uses_db: None, mcp_server: _McpServer) -> None:
    t = pxt.create_table('parallel_test', {'value': pxt.String})
    for i, (expr, message) in enumerate(
        [
            (parallel.search(' '), 'query must not be empty'),
            (parallel.fetch('file:///etc/passwd'), 'HTTP or HTTPS URL'),
            (parallel.fetch('https://example.org', objective='x' * 201), 'at most 200'),
            (parallel.search('query', session_id=''), '1 to 100'),
        ]
    ):
        t.add_computed_column(**{f'result_{i}': expr})
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=message):
            t.insert(value='trigger')
        t.drop_column(f'result_{i}')
    assert not mcp_server.requests


def test_timeout_and_cancellation(uses_db: None, mcp_server: _McpServer, monkeypatch: pytest.MonkeyPatch) -> None:
    # Coroutine cancellation has no synchronous table API; use the UDF's normal async execution entrypoint.
    async def run() -> None:
        mcp_server.block = True
        monkeypatch.setattr(parallel, '_REQUEST_TIMEOUT', 0.05)
        with pytest.raises(TimeoutError):
            await parallel.search.aexec('test timeout')
        monkeypatch.setattr(parallel, '_REQUEST_TIMEOUT', 60.0)
        task = asyncio.create_task(parallel.search.aexec('test cancellation'))
        while len(mcp_server.calls) < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert mcp_server.closed == 2

    asyncio.run(run())
