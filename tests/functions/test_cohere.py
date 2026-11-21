import asyncio
import json
from types import TracebackType
from typing import Any, Callable, Self

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import type_system as ts
from pixeltable.functions import cohere
from pixeltable.runtime import get_runtime, reset_runtime

from ..utils import pxt_raises, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

_DIM = 64


class _FakeResponse:
    def __init__(self, status: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        pass


class _FakeSession:
    """Stands in for the client's aiohttp session: records the requests and answers them with `handler`."""

    def __init__(self, handler: Callable[[str, dict[str, Any]], _FakeResponse]) -> None:
        self.handler = handler
        self.requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def post(self, endpoint: str, *, json: dict[str, Any], headers: dict[str, str]) -> _FakeResponse:
        self.requests.append((endpoint, json, headers))
        return self.handler(endpoint, json)


def _one_hot(text: str, dim: int) -> list[float]:
    """A unit vector that is the same for texts of the same length, so that similarity search has a known answer."""
    vector = [0.0] * dim
    vector[len(text) % dim] = 1.0
    return vector


def _cohere_api(endpoint: str, payload: dict[str, Any]) -> _FakeResponse:
    """Answers like the Cohere v2 API."""
    if endpoint == '/v2/embed':
        dim = payload.get('output_dimension', _DIM)
        vectors = [_one_hot(text, dim) for text in payload['texts']]
        return _FakeResponse(200, json.dumps({'id': 'e1', 'embeddings': {'float': vectors}, 'meta': {}}))
    assert endpoint == '/v2/rerank'
    # rank by document length, longest first
    order = sorted(range(len(payload['documents'])), key=lambda i: -len(payload['documents'][i]))
    results = [{'index': i, 'relevance_score': 1.0 / (rank + 1)} for rank, i in enumerate(order)]
    if 'top_n' in payload:
        results = results[: payload['top_n']]
    return _FakeResponse(
        200, json.dumps({'id': 'r1', 'results': results, 'meta': {'billed_units': {'search_units': 1}}})
    )


def _fake_session(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[str, dict[str, Any]], _FakeResponse]
) -> _FakeSession:
    session = _FakeSession(handler)
    monkeypatch.setattr(cohere._CohereClient, '_get_session', lambda self: session)
    return session


@pytest.fixture
def cohere_api(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    monkeypatch.setenv('COHERE_API_KEY', 'test-key')
    return _fake_session(monkeypatch, _cohere_api)


class TestCohereClient:
    def test_client_closed_on_reset(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('COHERE_API_KEY', 'test-key')
        client = get_runtime().get_client('cohere')

        # the session opens on the first request, on the runtime's loop; aiohttp wants it created in a coroutine
        async def open_session() -> Any:  # noqa: RUF029
            return client._get_session()

        session = get_runtime().event_loop.run_until_complete(open_session())
        reset_runtime()
        assert session.closed

    @pytest.mark.parametrize(
        ('status', 'headers', 'code', 'retry_after'),
        [
            (400, {}, pxt.ErrorCode.PROVIDER_BAD_REQUEST, None),
            (401, {}, pxt.ErrorCode.PROVIDER_AUTH_ERROR, None),
            (498, {}, pxt.ErrorCode.PROVIDER_AUTH_ERROR, None),
            (429, {'Retry-After': '7'}, pxt.ErrorCode.RATE_LIMITED, 7.0),
            (429, {'Retry-After': 'soon'}, pxt.ErrorCode.RATE_LIMITED, None),
            (500, {}, pxt.ErrorCode.PROVIDER_ERROR, None),
            (503, {'Retry-After': '2'}, pxt.ErrorCode.PROVIDER_OVERLOADED, 2.0),
        ],
    )
    def test_status_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        headers: dict[str, str],
        code: pxt.ErrorCode,
        retry_after: float | None,
    ) -> None:
        body = json.dumps({'message': f'error {status}'})
        _fake_session(monkeypatch, lambda endpoint, payload: _FakeResponse(status, body, headers))
        client = cohere._CohereClient(api_key='test-key')
        with pxt_raises(code, match=f'/v2/embed failed with status {status}: error {status}') as info:
            asyncio.run(client._post('/v2/embed', payload={}))
        assert isinstance(info.value, pxt.ExternalServiceError)
        assert info.value.provider == 'cohere'
        assert info.value.provider_http_status_code == status
        assert info.value.retry_after == retry_after

    def test_error_without_json_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_session(monkeypatch, lambda endpoint, payload: _FakeResponse(502, 'Bad Gateway'))
        client = cohere._CohereClient(api_key='test-key')
        with pxt_raises(pxt.ErrorCode.PROVIDER_ERROR, match='failed with status 502: Bad Gateway'):
            asyncio.run(client._post('/v2/rerank', payload={}))

    @pytest.mark.parametrize('body', ['not json', '[1, 2]'])
    def test_response_not_a_json_object(self, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
        _fake_session(monkeypatch, lambda endpoint, payload: _FakeResponse(200, body))
        client = cohere._CohereClient(api_key='test-key')
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='not a JSON object'):
            asyncio.run(client._post('/v2/embed', payload={}))


class TestCohere:
    def test_embeddings(self, cohere_api: _FakeSession) -> None:
        t = pxt.create_table('test_tbl', {'text': pxt.String})
        t.add_computed_column(
            embed=cohere.embeddings(t.text, model='embed-v4.0', input_type='search_document', output_dimension=_DIM)
        )
        assert isinstance(t.embed.col_type, ts.ArrayType)
        assert t.embed.col_type.shape == (_DIM,)
        texts = ['x' * n for n in range(1, 101)]
        validate_update_status(t.insert({'text': text} for text in texts), expected_rows=len(texts))

        res = t.select(t.text, t.embed).collect()
        for text, embed in zip(res['text'], res['embed']):
            assert embed.dtype == np.float32
            assert np.array_equal(embed, np.array(_one_hot(text, _DIM), dtype=np.float32))

        # Cohere accepts at most 96 texts per request
        assert all(len(payload['texts']) <= 96 for _, payload, _ in cohere_api.requests)
        assert sorted(text for _, payload, _ in cohere_api.requests for text in payload['texts']) == sorted(texts)
        endpoint, payload, headers = cohere_api.requests[0]
        assert endpoint == '/v2/embed'
        assert headers['Authorization'] == 'Bearer test-key'
        assert set(payload) == {'model', 'input_type', 'texts', 'embedding_types', 'output_dimension'}
        assert payload['model'] == 'embed-v4.0'
        assert payload['input_type'] == 'search_document'
        assert payload['embedding_types'] == ['float']
        assert payload['output_dimension'] == _DIM

    def test_embeddings_truncate(self, cohere_api: _FakeSession) -> None:
        t = pxt.create_table('test_tbl', {'text': pxt.String})
        t.add_computed_column(
            embed=cohere.embeddings(t.text, model='embed-english-v3.0', input_type='clustering', truncate='START')
        )
        validate_update_status(t.insert(text='hello'), expected_rows=1)
        _, payload, _ = cohere_api.requests[0]
        assert payload['truncate'] == 'START'
        assert 'output_dimension' not in payload

    @pytest.mark.parametrize(
        ('model', 'output_dimension', 'shape'),
        [
            ('embed-v4.0', None, (1536,)),
            ('embed-v4.0', 256, (256,)),
            ('embed-english-v3.0', None, (1024,)),
            ('embed-multilingual-light-v3.0', None, (384,)),
            ('some-future-model', None, (None,)),
        ],
    )
    def test_return_type(
        self, cohere_api: _FakeSession, model: str, output_dimension: int | None, shape: tuple[int | None]
    ) -> None:
        t = pxt.create_table('test_tbl', {'text': pxt.String})
        t.add_computed_column(
            embed=cohere.embeddings(t.text, model=model, input_type='search_query', output_dimension=output_dimension)
        )
        assert isinstance(t.embed.col_type, ts.ArrayType)
        assert t.embed.col_type.shape == shape

    def test_embedding_index(self, cohere_api: _FakeSession) -> None:
        # documents and queries are embedded with different input types: the index is on the column of document
        # embeddings, and the query side is embedded with input_type='search_query'
        t = pxt.create_table('test_tbl', {'text': pxt.String})
        t.add_computed_column(
            embed=cohere.embeddings(t.text, model='embed-v4.0', input_type='search_document', output_dimension=_DIM)
        )
        t.add_embedding_index(
            'embed',
            string_embed=cohere.embeddings.using(model='embed-v4.0', input_type='search_query', output_dimension=_DIM),
        )
        validate_update_status(t.insert([{'text': 'a'}, {'text': 'bbb'}, {'text': 'cc'}]), expected_rows=3)
        cohere_api.requests.clear()

        sim = t.embed.similarity(string='zz')
        res = t.order_by(sim, asc=False).limit(1).select(t.text).collect()
        assert res['text'] == ['cc']
        assert [payload['input_type'] for _, payload, _ in cohere_api.requests] == ['search_query']

    def test_embeddings_count_mismatch(self, cohere_api: _FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({'embeddings': {'float': [[0.5] * _DIM]}})
        _fake_session(monkeypatch, lambda endpoint, payload: _FakeResponse(200, body))
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='1 float embeddings for 2 input texts'):
            asyncio.run(cohere.embeddings.py_fn(['a', 'b'], model='embed-v4.0', input_type='search_document'))

    def test_rerank(self, cohere_api: _FakeSession) -> None:
        t = pxt.create_table('test_tbl', {'query': pxt.String, 'docs': pxt.Json})
        t.add_computed_column(reranked=cohere.rerank(t.query, t.docs, model='rerank-v3.5', top_n=2))
        docs = ['short', 'the longest document', 'medium doc']
        validate_update_status(t.insert(query='which is longest?', docs=docs), expected_rows=1)

        reranked = t.select(t.reranked).collect()['reranked'][0]
        assert [r['index'] for r in reranked['results']] == [1, 2]
        assert reranked['results'][0]['relevance_score'] == pytest.approx(1.0)
        assert reranked['meta'] == {'billed_units': {'search_units': 1}}

        endpoint, payload, _ = cohere_api.requests[0]
        assert endpoint == '/v2/rerank'
        assert payload == {'model': 'rerank-v3.5', 'query': 'which is longest?', 'documents': docs, 'top_n': 2}

    def test_rerank_unexpected_response(self, cohere_api: _FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({'results': [{'index': 0}]})
        _fake_session(monkeypatch, lambda endpoint, payload: _FakeResponse(200, body))
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='unexpected rerank response'):
            asyncio.run(cohere.rerank.py_fn('query', ['doc'], model='rerank-v3.5'))
