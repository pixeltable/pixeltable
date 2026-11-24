import contextlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.functions import keenable
from pixeltable.runtime import reset_runtime

from ..utils import pxt_raises, rerun_on_network_error, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

_QUERY = 'Pixeltable Python library'
_URL = 'https://pixeltable.com/'
_MISSING_URL = 'https://pixeltable.com/no-such-page-6f3a1c'

_SEARCH_RESPONSE = {
    'query': _QUERY,
    'mode': 'pro',
    'results': [{'url': _URL, 'title': 'Pixeltable', 'description': '', 'snippet': 'A snippet.'}],
}


@contextlib.contextmanager
def _configure(tmp_path: Path, api_key: str | None) -> Iterator[pytest.MonkeyPatch]:
    """Reconfigure the `keenable` client, from a config file of the test's own so the developer's supplies nothing."""
    project_root = Config.get().project_root
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('PIXELTABLE_CONFIG', str(tmp_path / 'config.toml'))
            if api_key is None:
                mp.delenv('KEENABLE_API_KEY', raising=False)
            else:
                mp.setenv('KEENABLE_API_KEY', api_key)
            Config.init(reinit=True, project_root=project_root)
            reset_runtime()
            yield mp
    finally:
        reset_runtime()
        Config.init(reinit=True, project_root=project_root)


@pytest.fixture
def anonymous(uses_db: None, tmp_path: Path) -> Iterator[None]:
    """Leave the client without an api_key, so that the free endpoints serve the test."""
    with _configure(tmp_path, None):
        yield


@pytest.fixture
def authenticated(uses_db: None) -> None:
    if not os.environ.get('KEENABLE_API_KEY'):
        pytest.skip('`KEENABLE_API_KEY` is not set.')


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI response calling one tool, in place of a model's, so that the tool tests need no LLM."""
    return {'choices': [{'message': {'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}


@pytest.mark.expensive
@pytest.mark.remote_api
@rerun_on_network_error()
def test_free_tier(anonymous: None) -> None:
    from pixeltable.functions.openai import invoke_tools

    tools = pxt.tools(keenable.search, keenable.fetch)
    t = pxt.create_table(
        'test_tbl', {'id': pxt.Int, 'query': pxt.String | None, 'url': pxt.String | None, 'response': pxt.Json}
    )
    t.add_computed_column(sources=keenable.search(t.query, max_results=3))
    t.add_computed_column(page=keenable.fetch(t.url, max_chars=2000))
    t.add_computed_column(tool_calls=invoke_tools(tools, t.response))

    rows = [
        {'id': 0, 'query': _QUERY, 'url': _URL, 'response': _tool_call('search', {'query': _QUERY})},
        # null inputs skip the direct calls, so this row makes only the fetch tool call
        {'id': 1, 'response': _tool_call('fetch', {'url': _URL})},
    ]
    validate_update_status(t.insert(rows), 2)
    first, second = t.order_by(t.id).collect()

    sources = first['sources']
    assert 0 < len(sources['results']) <= 3
    assert all(result['url'].startswith('http') for result in sources['results'])

    page = first['page']
    assert 'pixeltable' in page['content'].lower()
    assert len(page['content']) <= 2000 or page['truncated']

    assert first['tool_calls']['fetch'] is None
    assert len(first['tool_calls']['search'][0]['results']) > 0
    assert second['tool_calls']['search'] is None
    assert 'pixeltable' in second['tool_calls']['fetch'][0]['content'].lower()


@pytest.mark.expensive
@pytest.mark.remote_api
@rerun_on_network_error()
def test_missing_page(anonymous: None) -> None:
    t = pxt.create_table('test_tbl', {'url': pxt.String | None})
    t.add_computed_column(page=keenable.fetch(t.url))
    with pxt_raises(pxt.ErrorCode.PROVIDER_BAD_REQUEST, match='404'):
        t.insert(url=_MISSING_URL)


@pytest.mark.expensive
@pytest.mark.remote_api
@rerun_on_network_error()
class TestKeenable:
    def test_search_with_key(self, authenticated: None) -> None:
        t = pxt.create_table('test_tbl', {'query': pxt.String | None})
        t.add_computed_column(sources=keenable.search(t.query, max_results=3, site='github.com'))

        validate_update_status(t.insert(query=_QUERY), 1)
        response = t.select(t.sources).collect()['sources'][0]
        assert 0 < len(response['results']) <= 3
        assert all('github.com' in result['url'] for result in response['results'])

    def test_rejected_key(self, uses_db: None, tmp_path: Path) -> None:
        with _configure(tmp_path, 'not-a-key'):
            t = pxt.create_table('test_tbl', {'query': pxt.String | None})
            t.add_computed_column(sources=keenable.search(t.query))
            with pxt_raises(pxt.ErrorCode.PROVIDER_AUTH_ERROR, match='KEENABLE_API_KEY'):
                t.insert(query=_QUERY)


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
def rate_limited_service(uses_db: None, tmp_path: Path) -> Iterator[list[tuple[str, int]]]:
    """A stand-in for the free search endpoint that rejects the first request with a 429.

    Yields the paths and statuses it served, in order.
    """
    served: list[tuple[str, int]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers['Content-Length']))
            status = 429 if len(served) == 0 else 200
            if self.headers.get('X-Keenable-Title') is None:
                status = 400
            served.append((self.path, status))
            body = json.dumps({'error': 'rate limited'} if status != 200 else _SEARCH_RESPONSE).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with _serving(Handler) as port, _configure(tmp_path, None) as mp:
        mp.setattr(keenable, '_BASE_URL', f'http://127.0.0.1:{port}')
        yield served


def test_rate_limit_is_retried(rate_limited_service: list[tuple[str, int]]) -> None:
    t = pxt.create_table('test_tbl', {'query': pxt.String | None})
    t.add_computed_column(sources=keenable.search(t.query))

    validate_update_status(t.insert(query=_QUERY), 1)
    assert rate_limited_service == [('/v1/search/public', 429), ('/v1/search/public', 200)]
    assert t.select(t.sources).collect()['sources'][0]['results'][0]['url'] == _URL



def test_fetch_posts_urls(uses_db: None, tmp_path: Path) -> None:
    """The fetch UDF POSTs {urls, content} and returns results[0], not the GET query-string form."""
    served: list[tuple[str, str, dict[str, Any]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers['Content-Length']))
            payload = json.loads(raw.decode())
            served.append((self.path, self.command, payload))
            page = {
                'url': _URL,
                'title': 'Pixeltable',
                'content': 'Hello',
                'truncated': False,
                'total_chars': 5,
            }
            status = 200
            if self.headers.get('X-Keenable-Title') is None:
                status = 400
                body = json.dumps({'error': 'missing title'}).encode()
            else:
                body = json.dumps({'results': [page]}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with _serving(Handler) as port, _configure(tmp_path, None) as mp:
        mp.setattr(keenable, '_BASE_URL', f'http://127.0.0.1:{port}')
        t = pxt.create_table('test_tbl', {'url': pxt.String | None})
        t.add_computed_column(page=keenable.fetch(t.url, max_chars=2000))
        validate_update_status(t.insert(url=_URL), 1)

    assert served == [('/v1/fetch/public', 'POST', {'urls': [_URL], 'content': {'max_chars': 2000}})]
    page = t.select(t.page).collect()['page'][0]
    assert page['content'] == 'Hello'
    assert page['url'] == _URL
