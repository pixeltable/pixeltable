"""Exercise the sample's billing boundaries and API without paid provider calls."""

import asyncio
import base64
import pathlib
import shutil
import sys
import types
import uuid
from typing import Any, Callable, Iterator

import httpx
import pytest

import pixeltable as pxt
from pixeltable.utils.app_module import _evict_project_modules, load_app_module
from pixeltable.utils.http import is_retryable_error

from ..utils import DatabaseRoot, pxt_raises, skip_test_if_not_installed

_SAMPLE = pathlib.Path(__file__).parents[2] / 'docs/sample-apps/lead-enrichment'
_AUDIO = pathlib.Path(__file__).parents[1] / 'data/audio/sample.mp3'


@pytest.fixture
def sample_app(
    init_env: None, project_env: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[types.ModuleType]:
    monkeypatch.delenv('LEAD_ENRICHMENT_DEMO', raising=False)
    skip_test_if_not_installed('fastapi')
    skip_test_if_not_installed('openai')
    for name in ('app.py', 'udfs.py', 'ui.html'):
        shutil.copy(_SAMPLE / name, project_env / name)
    monkeypatch.syspath_prepend(str(project_env))
    try:
        yield load_app_module(str(project_env / 'app.py'), subject='application')
    finally:
        _evict_project_modules()


@pytest.fixture
def treg_transport(sample_app: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    client_type = httpx.AsyncClient
    monkeypatch.setenv('TREG_TOKEN', 'test-token')

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        monkeypatch.setattr(
            httpx, 'AsyncClient', lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs)
        )

    return install


def test_idempotency_and_audio_receipt(sample_app: types.ModuleType, treg_transport: Callable[..., None]) -> None:
    udfs = sys.modules['udfs']
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=_AUDIO.read_bytes(),
            headers={
                'Content-Type': 'audio/mpeg',
                'X-Treg-Cost-Micro': '0',
                'X-Treg-Original-Cost-Micro': '1234',
                'X-Treg-Idempotent-Replay': 'true',
                'X-Treg-Call-Id': 'voice-call',
            },
        )

    treg_transport(handler)
    lead_id = uuid.uuid4()
    prefix = 'a' * 120
    receipts = [
        asyncio.run(udfs.treg_voice_note.py_fn(row_id, script))
        for row_id, script in (
            (lead_id, prefix + 'first'),
            (lead_id, prefix + 'first'),
            (lead_id, prefix + 'second'),
            (uuid.uuid4(), prefix + 'first'),
        )
    ]
    keys = [r.headers['Idempotency-Key'] for r in requests]
    assert keys[0] == keys[1]
    assert len(set(keys)) == 3
    assert all(r.headers['model'] == 's2.1-pro' for r in requests)
    assert all(receipt['cost_micro'] == 1234 for receipt in receipts)
    assert base64.b64decode(receipts[0]['audio_base64']) == _AUDIO.read_bytes()
    assert pathlib.Path(udfs.voice_note_audio.py_fn(receipts[0])).read_bytes() == _AUDIO.read_bytes()


@pytest.mark.parametrize(
    'pages,expected',
    [
        (
            [{'markdown': 'First'}, {'text': 'Second'}, {'raw_content': 'Third'}, {'excerpts': ['Fourth', 'Fifth']}],
            'First\n\nSecond\n\nThird\n\nFourth\nFifth',
        ),
        ([], ''),
    ],
)
def test_site_pages(
    sample_app: types.ModuleType, treg_transport: Callable[..., None], pages: list[dict], expected: str
) -> None:
    treg_transport(
        lambda request: httpx.Response(
            200,
            json={'output': {'pages': pages}},
            headers={
                'X-Treg-Cost-Micro': '0',
                'X-Treg-Original-Cost-Micro': '2000',
                'X-Treg-Idempotent-Replay': 'true',
            },
        )
    )
    result = asyncio.run(sys.modules['udfs'].treg_extract_site.py_fn(uuid.uuid4(), 'example.com'))
    assert result['markdown'] == expected
    assert result['cost_micro'] == 2000


@pytest.mark.parametrize(
    'status,retryable',
    [
        (401, False),
        (402, False),
        (409, True),
        (422, False),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
def test_status_retry(
    sample_app: types.ModuleType, treg_transport: Callable[..., None], status: int, retryable: bool
) -> None:
    treg_transport(lambda request: httpx.Response(status, text='secret diagnostic must not reach the client'))
    with pytest.raises(pxt.ExternalServiceError, match=f'treg returned HTTP {status}') as error:
        asyncio.run(sys.modules['udfs']._treg_post('test', {}, lead_id=uuid.uuid4(), max_cost_usd=0.02))
    assert is_retryable_error(error.value)[0] is retryable
    assert 'secret diagnostic' not in str(error.value)


@pytest.mark.parametrize(
    'failure,code',
    [(httpx.ReadTimeout, pxt.ErrorCode.PROVIDER_TIMEOUT), (httpx.ConnectError, pxt.ErrorCode.PROVIDER_ERROR)],
)
def test_transport_retry(
    sample_app: types.ModuleType,
    treg_transport: Callable[..., None],
    failure: type[httpx.TransportError],
    code: pxt.ErrorCode,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure('test failure', request=request)

    treg_transport(handler)
    with pxt_raises(code, match='treg .* on test') as error:
        asyncio.run(sys.modules['udfs']._treg_post('test', {}, lead_id=uuid.uuid4(), max_cost_usd=0.02))
    assert error.value.error_code == code
    assert is_retryable_error(error.value)[0]


@pytest.mark.parametrize(
    'terminal,max_age,error_code',
    [
        ('done', 1.0, None),
        ('failed', 1.0, pxt.ErrorCode.PROVIDER_BAD_REQUEST),
        ('working', 0.0, pxt.ErrorCode.PROVIDER_TIMEOUT),
    ],
)
def test_pending_polls_existing_task(
    sample_app: types.ModuleType,
    treg_transport: Callable[..., None],
    terminal: str,
    max_age: float,
    error_code: pxt.ErrorCode | None,
) -> None:
    udfs = sys.modules['udfs']
    seen: list[str] = []
    descriptor = {
        'poll': {'endpoint': 'provider.status', 'param': {'name': 'id'}},
        'status': {'path': 'status', 'success': ['done'], 'failure': ['failed']},
        'result': {'path': 'result'},
        'interval': 0.001,
        'max_age': max_age,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.method == 'POST':
            return httpx.Response(
                202,
                json={
                    'output': {'pages': None},
                    '_treg': {
                        'outcome': 'pending',
                        'served_by': 'provider.extract',
                        'call_ref': 'child-call',
                        'async': {'task_id': 'task-one', 'poll': descriptor['poll'], 'status': descriptor['status']},
                    },
                },
                headers={'X-Treg-Cost-Micro': '100'},
            )
        assert 'Idempotency-Key' not in request.headers
        if '/catalog/' in request.url.path:
            return httpx.Response(200, json={'endpoint': {'async': descriptor}})
        if request.url.path == '/calls/child-call':
            return httpx.Response(200, json={'charged_micro': 2000})
        assert request.url.params['id'] == 'task-one'
        return httpx.Response(200, json={'status': terminal, 'result': {'pages': [{'markdown': 'Completed page'}]}})

    treg_transport(handler)
    if error_code is None:
        result = asyncio.run(udfs.treg_extract_site.py_fn(uuid.uuid4(), 'example.com'))
        assert result['markdown'] == 'Completed page'
        assert result['cost_micro'] == 2100
    else:
        with pxt_raises(error_code, match='treg task child-call'):
            asyncio.run(udfs.treg_extract_site.py_fn(uuid.uuid4(), 'example.com'))
    assert seen.count('/call/treg.web.extract') == 1
    assert seen.count('/call/provider.status') == (0 if max_age == 0 else 1)


@pytest.mark.parametrize(
    'verdict,role,expected',
    [
        ('valid', 'Co-Founder & CTO', 'priority'),
        ('valid', 'VP Engineering', 'priority'),
        ('valid', 'Head of Sales', 'priority'),
        ('valid', 'Doctor', 'ready'),  # 'cto' must not match inside 'Doctor'
        ('valid', 'Contractor', 'ready'),  # 'cto' must not match inside 'Contractor'
        ('valid', 'Headhunter', 'ready'),  # 'head' must not match inside 'Headhunter'
        ('valid', 'Engineer', 'ready'),
        ('valid', None, 'ready'),
        ('catch_all', 'CEO', 'review'),
        ('risky', 'CEO', 'review'),
        ('unknown', None, 'review'),
        ('invalid', 'CEO', 'skip'),
        (None, 'CEO', 'skip'),
    ],
)
def test_lead_tier(sample_app: types.ModuleType, verdict: str | None, role: str | None, expected: str) -> None:
    assert sys.modules['udfs'].lead_tier.py_fn(verdict, role) == expected


@pytest.mark.db_roots('local', reason='Runs the sample with mocked providers in process')
def test_sample_api(sample_app: types.ModuleType, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
    from pixeltable.runtime import reset_runtime
    from pixeltable.serving import _fastapi

    run_endpoint = _fastapi._run_endpoint_op

    def run_and_close(*args: Any, **kwargs: Any) -> Any:
        try:
            return run_endpoint(*args, **kwargs)
        finally:
            # Foreground ASGI workers own thread-local runtimes beyond TestClient's lifespan.
            reset_runtime()

    monkeypatch.setattr(_fastapi, '_run_endpoint_op', run_and_close)
    udfs = sys.modules['udfs']
    calls: list[str] = []
    fail_site = False

    async def post(endpoint: str, body: dict[str, Any], **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        calls.append(endpoint)
        if endpoint == 'treg.people.email.find':
            return udfs._TregResult(
                output={'email': 'test@example.com', 'verified': True},
                content=b'',
                content_type='application/json',
                cost_micro=1000,
                call_id=None,
                served_by=None,
                verification={'verdict': 'valid'},
            )
        if endpoint == 'treg.web.extract':
            if fail_site:
                raise pxt.ExternalServiceError(pxt.ErrorCode.PROVIDER_BAD_REQUEST, 'Mock site failure')
            return udfs._TregResult(
                output={'pages': [{'markdown': 'Real company context'}]},
                content=b'',
                content_type='application/json',
                cost_micro=2000,
                call_id=None,
                served_by=None,
                verification=None,
            )
        return udfs._TregResult(
            output=None,
            content=_AUDIO.read_bytes(),
            content_type='audio/mpeg',
            cost_micro=3000,
            call_id=None,
            served_by=None,
            verification=None,
        )

    async def chat(*args: Any, **kwargs: Any) -> dict:
        await asyncio.sleep(0)
        return {'choices': [{'message': {'content': 'Hello from the sample'}}]}

    monkeypatch.setattr(udfs, '_treg_post', post)
    monkeypatch.setattr(udfs.chat_completions, 'py_fns', [chat])
    sample_app.TableModel.create_all(db_root.make_catalog_path(''))
    router = sample_app.leads_api
    router.bind(db_root.make_catalog_path(''))
    table = pxt.get_table(db_root.make_catalog_path('leads'))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    api = FastAPI()
    api.include_router(router)
    with TestClient(api) as client:
        payload = {
            'lead_id': str(uuid.uuid4()),
            'full_name': 'Test Person',
            'domain': 'example.com',
            'role': 'Founder',
            'stage': 'new',
        }
        preview = client.post('/preview', json=payload)
        assert preview.status_code == 200, preview.text
        assert preview.json()['cost_usd'] == pytest.approx(0.006)
        assert table.count() == 0
        for name in ('One', 'Two'):
            payload.update(full_name=name, lead_id=str(uuid.uuid4()))
            response = client.post('/leads', json=payload)
            assert response.status_code == 200, response.text
            assert response.json()['cost_usd'] == pytest.approx(0.006)
        assert len(client.get('/leads/by-domain', params={'domain': 'example.com'}).json()['rows']) == 2
        assert len(client.get('/leads/all').json()['rows']) == 2
        assert len(client.get('/leads/by-tier', params={'tier': 'priority', 'stage': 'new'}).json()['rows']) == 2
        audio = client.get('/leads/voice-note', params={'lead_id': payload['lead_id']})
        assert audio.status_code == 200, audio.text
        assert audio.content == _AUDIO.read_bytes()
        calls_before = len(calls)
        changed = client.post('/leads/stage', json={'lead_id': payload['lead_id'], 'stage': 'queued'})
        assert changed.status_code == 200, changed.text
        assert len(calls) == calls_before
        deleted = client.post('/leads/delete', json={'lead_id': payload['lead_id']})
        assert deleted.status_code == 200, deleted.text
        assert table.count() == 1
        table.recompute_columns('voice_note')
        assert len(calls) == calls_before
        fail_site = True
        failed_payload = {**payload, 'lead_id': str(uuid.uuid4())}
        failed = client.post('/leads', json=failed_payload)
        assert failed.status_code == 400, failed.text
        assert table.count() == 1
        # Only on_error='ignore' leaves cells for errors-only recovery.
        saved = table.insert([{**failed_payload, 'lead_id': uuid.UUID(failed_payload['lead_id'])}], on_error='ignore')
        assert saved.num_excs > 0
        assert table.count() == 2
        fail_site = False
        recovered = table.recompute_columns('site', errors_only=True)
        assert recovered.num_excs == 0
        assert table.select(table.site_markdown).collect()['site_markdown'] == ['Real company context'] * 2


def test_pending_mailbox_verification(sample_app: types.ModuleType, treg_transport: Callable[..., None]) -> None:
    """BounceBan's native terminal result and ledger settle the original verification call."""
    seen: list[str] = []
    descriptor = {
        'poll': {'endpoint': 'bounceban.people.email.verify.status', 'param': {'name': 'id'}},
        'status': {'path': 'status', 'success': ['success']},
        'result': {'path': 'result'},
        'interval': 0.001,
        'max_age': 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.method == 'POST':
            return httpx.Response(
                200,
                json={
                    'output': {'email': 'test@example.com', 'verified': True},
                    '_treg': {
                        'verification': {
                            'verdict': 'unknown',
                            'pending': True,
                            'served_by': 'bounceban.people.email.verify',
                            'call_id': 'verify-call',
                        }
                    },
                },
                headers={
                    'X-Treg-Cost-Micro': '0',
                    'X-Treg-Idempotent-Replay': 'true',
                    'X-Treg-Original-Cost-Micro': '1000',
                },
            )
        if '/catalog/' in request.url.path:
            return httpx.Response(200, json={'endpoint': {'async': descriptor}})
        if request.url.path == '/calls/verify-call':
            return httpx.Response(200, json={'async_task': {'task_id': 'bb-1'}, 'charged_micro': 500})
        assert request.url.path == '/call/bounceban.people.email.verify.status'
        assert request.url.params['id'] == 'bb-1'
        return httpx.Response(200, json={'id': 'bb-1', 'status': 'success', 'result': 'deliverable'})

    treg_transport(handler)
    result = asyncio.run(sys.modules['udfs'].treg_find_email.py_fn(uuid.uuid4(), 'Test Person', 'example.com'))
    assert result['verdict'] == 'valid'
    assert result['cost_micro'] == 1500
    assert seen.count('/call/treg.people.email.find') == 1
    assert seen.count('/call/bounceban.people.email.verify.status') == 1


@pytest.mark.db_roots('local', reason='Validates the offline sample without provider calls')
def test_explicit_demo(sample_app: types.ModuleType, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
    udfs = sys.modules['udfs']
    monkeypatch.delenv('TREG_TOKEN', raising=False)
    with pxt_raises(pxt.ErrorCode.MISSING_CREDENTIALS, match='TREG_TOKEN'):
        asyncio.run(udfs.treg_find_email.py_fn(uuid.uuid4(), 'Test Person', 'example.com'))
    monkeypatch.setenv('LEAD_ENRICHMENT_DEMO', '1')
    contact = asyncio.run(udfs.treg_find_email.py_fn(uuid.uuid4(), 'Test Person', 'example.com'))
    assert contact['email'] is None
    assert contact['verified'] is False
    assert contact['verdict'] == 'unknown'
    assert contact['cost_micro'] == 0
    site = asyncio.run(udfs.treg_extract_site.py_fn(uuid.uuid4(), 'example.com'))
    assert 'synthetic' in site['markdown']
    draft = asyncio.run(udfs.outreach_copywriter.py_fn('test prompt'))
    assert 'Demo' in draft['choices'][0]['message']['content']
    receipt = asyncio.run(udfs.treg_voice_note.py_fn(uuid.uuid4(), 'test script'))
    assert receipt['cost_micro'] == 0
    assert base64.b64decode(receipt['audio_base64']).startswith(b'RIFF')
    assert pathlib.Path(udfs.voice_note_audio.py_fn(receipt)).suffix == '.wav'
    assert 'Demo mode' in sample_app.lead_studio_ui().body.decode()
    sample_app.TableModel.create_all(db_root.make_catalog_path(''))
    table = pxt.get_table(db_root.make_catalog_path('leads'))
    table.insert(
        [
            {
                'lead_id': uuid.uuid4(),
                'full_name': 'Demo Person',
                'domain': 'example.com',
                'role': 'Founder',
                'stage': 'new',
            }
        ]
    )
    result = table.select(table.email, table.tier, table.cost_usd).collect()[0]
    assert result['email'] is None
    assert result['tier'] == 'review'
    assert result['cost_usd'] == 0
