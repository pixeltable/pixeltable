import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import httpx
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.config import Config
from pixeltable.runtime import get_runtime, reset_runtime
from pixeltable.utils.local_store import TempStore

from ..utils import (
    get_documents,
    get_image_files,
    get_video_files,
    pxt_raises,
    rerun_on_network_error,
    skip_test_if_no_client,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

_API = 'https://agent.vlm.run/v1'
_CHAT = f'POST {_API}/openai/chat/completions'
_ARTIFACTS = f'GET {_API}/artifacts'
_MODEL = 'vlmrun-orion-2:fast'


class _FakeVlmRun:
    """The VLM Run API, as served to the registered client: each request takes the next queued response for its
    method and URL, except uploads, which always succeed."""

    client: 'pxtf.vlmrun._VlmRunClient'
    requests: list[httpx.Request]
    responses: defaultdict[str, list[httpx.Response]]

    def __init__(self) -> None:
        self.client = pxtf.vlmrun._VlmRunClient('test-key', timeout=60.0, transport=httpx.MockTransport(self._handle))
        self.requests = []
        self.responses = defaultdict(list)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f'{request.method} {request.url.copy_with(query=None)}'
        if key == f'POST {_API}/files':
            return httpx.Response(201, json={'id': f'file-{len(self.requests_to(key))}'})
        if request.method == 'DELETE' and request.url.path.startswith('/v1/files/'):
            return httpx.Response(200, json={'id': request.url.path.rsplit('/', 1)[-1]})
        assert len(self.responses[key]) > 0, f'unexpected request: {key}'
        response = self.responses[key].pop(0) if len(self.responses[key]) > 1 else self.responses[key][0]
        # a new response each time, since a response's stream can be read only once
        return httpx.Response(response.status_code, headers=response.headers, stream=response.stream)

    def add(self, key: str, *responses: httpx.Response) -> None:
        """Queues responses for `key` (method and URL); the last one answers every further request."""
        self.responses[key].extend(responses)

    def requests_to(self, key: str) -> list[httpx.Request]:
        return [r for r in self.requests if f'{r.method} {r.url.copy_with(query=None)}' == key]

    def deleted_files(self) -> list[str]:
        return sorted(r.url.path.rsplit('/', 1)[-1] for r in self.requests if r.method == 'DELETE')


@pytest.fixture
def fake_vlmrun(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeVlmRun]:
    fake = _FakeVlmRun()
    monkeypatch.setattr(pxtf.vlmrun, '_vlmrun_client', lambda: fake.client)
    monkeypatch.setattr(pxtf.vlmrun, 'exponential_backoff', lambda attempt: 0.0)
    yield fake
    get_runtime().event_loop.run_until_complete(fake.client.close())


def _completion(content: Any, **fields: Any) -> httpx.Response:
    content = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(201, json={'choices': [{'message': {'role': 'assistant', 'content': content}}], **fields})


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def _image_bytes() -> bytes:
    return Path(get_image_files()[0]).read_bytes()


def _smallest_pdf() -> str:
    return min((d for d in get_documents() if d.endswith('.pdf')), key=os.path.getsize)


def _smallest_video() -> str:
    return min((v for v in get_video_files() if v.endswith('.mp4')), key=os.path.getsize)


class TestVLMRunClient:
    def test_client_closed_on_reset(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('VLMRUN_API_KEY', 'test-key')
        http_client = get_runtime().get_client('vlmrun')._http
        reset_runtime()
        assert http_client.is_closed

    def test_missing_credentials(self, uses_db: None, tmp_path: Path) -> None:
        project_root = Config.get().project_root
        try:
            with pytest.MonkeyPatch.context() as mp:
                # a config file of the test's own, so that the developer's config.toml can't supply an api_key
                mp.delenv('VLMRUN_API_KEY', raising=False)
                mp.setenv('PIXELTABLE_CONFIG', str(tmp_path / 'config.toml'))
                Config.init(reinit=True, project_root=project_root)
                reset_runtime()
                t = pxt.create_table('test_tbl', {'image': pxt.Image})
                messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': t.image}]}]
                t.add_computed_column(response=pxtf.vlmrun.chat_completions(messages, model=_MODEL))
                with pxt_raises(pxt.ErrorCode.MISSING_CREDENTIALS, match='VLMRUN_API_KEY'):
                    t.insert(image=get_image_files()[0])
        finally:
            reset_runtime()
            Config.init(reinit=True, project_root=project_root)


class TestVLMRunMocked:
    def test_chat_completions(self, fake_vlmrun: _FakeVlmRun) -> None:
        fake_vlmrun.add(_CHAT, _completion('a description', session_id='sess-1'))
        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'video': pxt.Video, 'doc': pxt.Document})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe these'},
                    {'type': 'image_url', 'image_url': t.image},
                    {'type': 'video_url', 'video_url': t.video},
                    {'type': 'file_url', 'file_url': t.doc},
                    {'type': 'image_url', 'image_url': 'https://example.com/a.jpg'},
                    {'type': 'video_url', 'video_url': {'url': 'https://example.com/v.mp4'}},
                ],
            }
        ]
        t.add_computed_column(
            response=pxtf.vlmrun.chat_completions(messages, model=_MODEL, model_kwargs={'toolsets': ['document']})
        )
        validate_update_status(t.insert(image=get_image_files()[0], video=_smallest_video(), doc=_smallest_pdf()), 1)

        response = t.select(t.response).collect()[0]['response']
        assert response['choices'][0]['message']['content'] == 'a description'
        assert response['session_id'] == 'sess-1'
        assert len(fake_vlmrun.requests_to(f'POST {_API}/files')) == 3
        body = _body(fake_vlmrun.requests_to(_CHAT)[0])
        assert body['model'] == _MODEL
        assert body['toolsets'] == ['document']
        assert body['messages'][0]['content'] == [
            {'type': 'text', 'text': 'Describe these'},
            {'type': 'input_file', 'file_id': 'file-1'},
            {'type': 'input_file', 'file_id': 'file-2'},
            {'type': 'input_file', 'file_id': 'file-3'},
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.jpg'}},
            {'type': 'video_url', 'video_url': {'url': 'https://example.com/v.mp4'}},
        ]
        assert all(r.headers['Authorization'] == 'Bearer test-key' for r in fake_vlmrun.requests)
        assert fake_vlmrun.deleted_files() == ['file-1', 'file-2', 'file-3']

    def test_long_running_request(self, fake_vlmrun: _FakeVlmRun) -> None:
        # the gateway redirects the request; its result URL returns no result, a run of 5xx and a redirect before the
        # result
        fake_vlmrun.add(_CHAT, httpx.Response(303, headers={'Location': '/v1/openai/results/1'}))
        fake_vlmrun.add(
            f'GET {_API}/openai/results/1',
            httpx.Response(204),
            httpx.Response(200),
            *[httpx.Response(500)] * 8,
            httpx.Response(303, headers={'Location': 'https://results.example.com/1'}),
        )
        fake_vlmrun.add('GET https://results.example.com/1', _completion('done'))
        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(
            response=pxtf.vlmrun.chat_completions([{'role': 'user', 'content': t.prompt}], model=_MODEL)
        )
        validate_update_status(t.insert(prompt='hi'), 1)

        assert t.select(t.response).collect()[0]['response']['choices'][0]['message']['content'] == 'done'
        assert len(fake_vlmrun.requests_to(f'GET {_API}/openai/results/1')) == 11
        # the API key goes to the API host only
        assert 'Authorization' not in fake_vlmrun.requests_to('GET https://results.example.com/1')[0].headers

    @pytest.mark.parametrize(
        ('response', 'error_code', 'match'),
        [
            (httpx.Response(200, content=b'<html>gateway</html>'), pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, 'JSON'),
            (httpx.Response(400, json={'detail': 'bad id'}), pxt.ErrorCode.PROVIDER_BAD_REQUEST, 'status 400: bad id'),
        ],
    )
    def test_long_running_request_errors(
        self, fake_vlmrun: _FakeVlmRun, response: httpx.Response, error_code: pxt.ErrorCode, match: str
    ) -> None:
        fake_vlmrun.add(_CHAT, httpx.Response(303, headers={'Location': '/v1/openai/results/1'}))
        fake_vlmrun.add(f'GET {_API}/openai/results/1', response)
        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(
            response=pxtf.vlmrun.chat_completions([{'role': 'user', 'content': t.prompt}], model=_MODEL)
        )
        with pxt_raises(error_code, match=match):
            t.insert(prompt='hi')
        assert len(fake_vlmrun.requests_to(f'GET {_API}/openai/results/1')) == 1

    def test_timeout(self, fake_vlmrun: _FakeVlmRun) -> None:
        fake_vlmrun.client.timeout = 0.2
        fake_vlmrun.add(_CHAT, httpx.Response(303, headers={'Location': '/v1/openai/results/1'}))
        fake_vlmrun.add(f'GET {_API}/openai/results/1', httpx.Response(204))
        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(
            response=pxtf.vlmrun.chat_completions([{'role': 'user', 'content': t.prompt}], model=_MODEL)
        )
        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match=r'did not complete within 0\.2s'):
            t.insert(prompt='hi')

    def test_generate_image(self, fake_vlmrun: _FakeVlmRun) -> None:
        # the artifact id comes with an extension; the artifact isn't there yet, then rate-limited, then empty, then
        # ready
        fake_vlmrun.add(_CHAT, _completion({'image': {'id': 'img_a1b2c3.png'}}, session_id='sess-1'))
        fake_vlmrun.add(
            _ARTIFACTS,
            httpx.Response(404),
            httpx.Response(429, headers={'Retry-After': '0'}),
            httpx.Response(200),
            httpx.Response(200, content=_image_bytes()),
        )
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(edited=pxtf.vlmrun.generate_image('Blur the faces', image=t.image, model=_MODEL))
        validate_update_status(t.insert(image=get_image_files()[0]), 1)

        edited = t.select(t.edited).collect()[0]['edited']
        assert isinstance(edited, PIL.Image.Image)
        assert edited.size == PIL.Image.open(get_image_files()[0]).size
        body = _body(fake_vlmrun.requests_to(_CHAT)[0])
        assert body['toolsets'] == ['image-gen']
        assert body['response_format']['schema']['required'] == ['image']
        assert body['messages'][1]['content'] == [
            {'type': 'text', 'text': 'Blur the faces'},
            {'type': 'input_file', 'file_id': 'file-1'},
        ]
        artifact_requests = fake_vlmrun.requests_to(_ARTIFACTS)
        assert len(artifact_requests) == 4
        assert dict(artifact_requests[0].url.params) == {'object_id': 'img_a1b2c3', 'session_id': 'sess-1'}

    @pytest.mark.parametrize(
        ('status_code', 'error_code'),
        [(403, pxt.ErrorCode.PROVIDER_AUTH_ERROR), (404, pxt.ErrorCode.PROVIDER_BAD_REQUEST)],
    )
    def test_artifact_errors(
        self, fake_vlmrun: _FakeVlmRun, monkeypatch: pytest.MonkeyPatch, status_code: int, error_code: pxt.ErrorCode
    ) -> None:
        monkeypatch.setattr(pxtf.vlmrun, '_NOT_FOUND_GRACE', 0.0)
        fake_vlmrun.add(_CHAT, _completion({'image': {'id': 'img_a1b2c3'}}, session_id='sess-1'))
        fake_vlmrun.add(_ARTIFACTS, httpx.Response(status_code, json={'error': {'message': 'no such artifact'}}))
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(annotated=pxtf.vlmrun.annotate_image('Box the people', t.image, model=_MODEL))
        with pxt_raises(error_code, match=f'status {status_code}: no such artifact'):
            t.insert(image=get_image_files()[0])
        assert len(fake_vlmrun.requests_to(_ARTIFACTS)) == 1
        assert _body(fake_vlmrun.requests_to(_CHAT)[0])['toolsets'] == ['viz']
        assert fake_vlmrun.deleted_files() == ['file-1']

    def test_generate_video(self, fake_vlmrun: _FakeVlmRun) -> None:
        # the body of a url_ artifact is the URL of the video
        video = _smallest_video()
        fake_vlmrun.add(_CHAT, _completion({'video': {'id': 'url_f41446'}}, session_id='sess-1'))
        fake_vlmrun.add(_ARTIFACTS, httpx.Response(200, text='https://cdn.example.com/v.mp4\n'))
        fake_vlmrun.add('GET https://cdn.example.com/v.mp4', httpx.Response(200, content=Path(video).read_bytes()))
        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'video': pxt.Video})
        t.add_computed_column(
            output=pxtf.vlmrun.generate_video('Continue it', image=t.image, video=t.video, model=_MODEL)
        )
        validate_update_status(t.insert(image=get_image_files()[0], video=video), 1)

        output = t.select(t.output).collect()[0]['output']
        assert Path(output).read_bytes() == Path(video).read_bytes()
        assert len(fake_vlmrun.requests_to(f'POST {_API}/files')) == 2
        assert 'Authorization' not in fake_vlmrun.requests_to('GET https://cdn.example.com/v.mp4')[0].headers

    def test_presigned_upload(self, fake_vlmrun: _FakeVlmRun, monkeypatch: pytest.MonkeyPatch) -> None:
        # an image or a file over the direct upload limit goes to a presigned storage URL
        monkeypatch.setattr(pxtf.vlmrun, '_DIRECT_UPLOAD_LIMIT', 0)
        video = _smallest_video()
        fake_vlmrun.add(
            f'POST {_API}/files/presigned-url',
            httpx.Response(
                201, json={'id': 'file-big', 'url': 'https://storage.example.com/up', 'content_type': 'video/mp4'}
            ),
        )
        fake_vlmrun.add('PUT https://storage.example.com/up', httpx.Response(200))
        fake_vlmrun.add(f'GET {_API}/files/verify-upload/file-big', httpx.Response(200, json={'id': 'file-big'}))
        fake_vlmrun.add(_CHAT, _completion({'video': 'https://cdn.example.com/v.mp4'}))
        fake_vlmrun.add('GET https://cdn.example.com/v.mp4', httpx.Response(200, content=Path(video).read_bytes()))
        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'video': pxt.Video})
        t.add_computed_column(
            output=pxtf.vlmrun.generate_video('Continue it', image=t.image, video=t.video, model=_MODEL)
        )
        validate_update_status(t.insert(image=get_image_files()[0], video=video), 1)

        assert fake_vlmrun.requests_to(f'POST {_API}/files') == []
        puts = fake_vlmrun.requests_to('PUT https://storage.example.com/up')
        assert len(puts) == 2
        assert all('Authorization' not in put.headers for put in puts)
        assert all(put.headers['Content-Length'] == str(len(put.content)) for put in puts)
        assert puts[1].content == Path(video).read_bytes()
        assert _body(fake_vlmrun.requests_to(_CHAT)[0])['messages'][1]['content'][1:] == [
            {'type': 'input_file', 'file_id': 'file-big'},
            {'type': 'input_file', 'file_id': 'file-big'},
        ]

    @pytest.mark.parametrize('source', ['redirect', 'url_artifact'])
    def test_non_https_url(self, fake_vlmrun: _FakeVlmRun, source: str) -> None:
        if source == 'redirect':
            fake_vlmrun.add(_CHAT, _completion({'document': 'https://cdn.example.com/d.pdf'}))
            fake_vlmrun.add(
                'GET https://cdn.example.com/d.pdf',
                httpx.Response(302, headers={'Location': 'http://169.254.169.254/latest/meta-data'}),
            )
        else:
            fake_vlmrun.add(_CHAT, _completion({'document': {'id': 'url_f41446'}}, session_id='sess-1'))
            fake_vlmrun.add(_ARTIFACTS, httpx.Response(200, text='http://169.254.169.254/latest/meta-data'))
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(redacted=pxtf.vlmrun.generate_document('Redact the names', t.doc, model=_MODEL))
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='not https: http://169.254.169.254'):
            t.insert(doc=_smallest_pdf())
        assert fake_vlmrun.requests_to('GET http://169.254.169.254/latest/meta-data') == []

    def test_failed_download_removed(self, fake_vlmrun: _FakeVlmRun) -> None:
        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield b'%PDF-partial'
                raise httpx.ReadError('stream broke')

        # a download that keeps breaking is retried until the deadline, which leaves no partial file
        fake_vlmrun.client.timeout = 0.5
        fake_vlmrun.add(_CHAT, _completion({'document': 'https://cdn.example.com/d.pdf'}))
        fake_vlmrun.add('GET https://cdn.example.com/d.pdf', httpx.Response(200, stream=BrokenStream()))
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(redacted=pxtf.vlmrun.generate_document('Redact the names', t.doc, model=_MODEL))
        tmp_count = TempStore.count()
        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='did not complete'):
            t.insert(doc=_smallest_pdf())
        assert TempStore.count() == tmp_count
        assert len(fake_vlmrun.requests_to('GET https://cdn.example.com/d.pdf')) > 1

    def test_generate_document(self, fake_vlmrun: _FakeVlmRun) -> None:
        pdf = _smallest_pdf()
        fake_vlmrun.add(_CHAT, _completion({'document': 'https://cdn.example.com/d.pdf'}))
        fake_vlmrun.add(
            'GET https://cdn.example.com/d.pdf',
            httpx.Response(503),
            httpx.Response(200, content=Path(pdf).read_bytes()),
        )
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(redacted=pxtf.vlmrun.generate_document('Redact the names', t.doc, model=_MODEL))
        validate_update_status(t.insert(doc=pdf), 1)

        redacted = t.select(t.redacted).collect()[0]['redacted']
        assert Path(redacted).read_bytes()[:5] == b'%PDF-'
        assert _body(fake_vlmrun.requests_to(_CHAT)[0])['toolsets'] == ['document']

    def test_unsupported_format(self, fake_vlmrun: _FakeVlmRun) -> None:
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        messages = [{'role': 'user', 'content': [{'type': 'file_url', 'file_url': t.doc}]}]
        t.add_computed_column(response=pxtf.vlmrun.chat_completions(messages, model=_MODEL))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'Unsupported file format: \.txt'):
            t.insert(doc=next(d for d in get_documents() if d.endswith('.txt')))
        assert fake_vlmrun.requests == []

    @pytest.mark.parametrize(
        ('col_type', 'value', 'match'),
        [(pxt.Int, 1, '`image_url` must be a media column'), (pxt.Image | None, None, '`image_url` is null')],
    )
    def test_invalid_media_value(self, fake_vlmrun: _FakeVlmRun, col_type: Any, value: Any, match: str) -> None:
        t = pxt.create_table('test_tbl', {'media': col_type})
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': t.media}]}]
        t.add_computed_column(response=pxtf.vlmrun.chat_completions(messages, model=_MODEL))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=match):
            t.insert(media=value)
        assert fake_vlmrun.requests == []

    @pytest.mark.parametrize('model_kwargs', [{'response_format': {'type': 'json_object'}}, {'toolsets': ['video']}])
    def test_reserved_model_kwargs(self, fake_vlmrun: _FakeVlmRun, model_kwargs: dict[str, Any]) -> None:
        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=pxtf.vlmrun.generate_image(t.prompt, model=_MODEL, model_kwargs=model_kwargs))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=f'cannot contain `{next(iter(model_kwargs))}`'):
            t.insert(prompt='A red circle')
        assert fake_vlmrun.requests == []


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestVLMRun:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'video': pxt.Video, 'doc': pxt.Document})

        def ask(text: str, *parts: dict, **kwargs: Any) -> pxt.exprs.Expr:
            content = [{'type': 'text', 'text': text}, *parts]
            return pxtf.vlmrun.chat_completions([{'role': 'user', 'content': content}], **kwargs)

        image_part = {'type': 'image_url', 'image_url': t.image}
        doc_part = {'type': 'file_url', 'file_url': t.doc}
        t.add_computed_column(text=ask('What is 2+2?', model=_MODEL))
        t.add_computed_column(image_fast=ask('Describe this image briefly', image_part, model=_MODEL))
        t.add_computed_column(image_auto=ask('Describe this image briefly', image_part, model='vlmrun-orion-2:auto'))
        t.add_computed_column(doc_title=ask('What is the title of this document?', doc_part, model=_MODEL))
        t.add_computed_column(
            doc_toolset=ask(
                'Extract all text from this document', doc_part, model=_MODEL, model_kwargs={'toolsets': ['document']}
            )
        )
        t.add_computed_column(
            video_description=ask(
                'Describe this video briefly', {'type': 'video_url', 'video_url': t.video}, model=_MODEL
            )
        )
        schema = {
            'type': 'object',
            'properties': {'title': {'type': 'string'}, 'num_pages': {'type': 'integer'}},
            'required': ['title', 'num_pages'],
        }
        t.add_computed_column(
            structured=ask(
                'Extract the title and the number of pages',
                doc_part,
                model=_MODEL,
                model_kwargs={'response_format': {'type': 'json_schema', 'schema': schema}},
            )
        )
        validate_update_status(t.insert(image=get_image_files()[0], video=_smallest_video(), doc=_smallest_pdf()), 1)

        row = t.collect()[0]
        columns = ('text', 'image_fast', 'image_auto', 'doc_title', 'doc_toolset', 'video_description', 'structured')
        assert all(len(row[col]['choices'][0]['message']['content']) > 0 for col in columns)
        assert '4' in row['text']['choices'][0]['message']['content']
        assert row['image_auto']['model'] == 'vlmrun-orion-2:auto'
        structured = json.loads(row['structured']['choices'][0]['message']['content'])
        assert isinstance(structured['title'], str) and isinstance(structured['num_pages'], int)

    def test_generate_image(self, uses_db: None) -> None:
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'image': pxt.Image})
        t.add_computed_column(generated=pxtf.vlmrun.generate_image(t.prompt, model=_MODEL))
        t.add_computed_column(edited=pxtf.vlmrun.generate_image('Blur all the faces', image=t.image, model=_MODEL))
        t.add_computed_column(
            annotated=pxtf.vlmrun.annotate_image('Draw bounding boxes around all objects', t.image, model=_MODEL)
        )
        validate_update_status(t.insert(prompt='A red circle on a white background', image=get_image_files()[0]), 1)

        row = t.collect()[0]
        assert all(isinstance(row[col], PIL.Image.Image) for col in ('generated', 'edited', 'annotated'))

    def test_generate_video(self, uses_db: None) -> None:
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'image': pxt.Image, 'video': pxt.Video})
        t.add_computed_column(from_text=pxtf.vlmrun.generate_video(t.prompt, model=_MODEL))
        t.add_computed_column(from_image=pxtf.vlmrun.generate_video(t.prompt, image=t.image, model=_MODEL))
        t.add_computed_column(from_video=pxtf.vlmrun.generate_video(t.prompt, video=t.video, model=_MODEL))
        for col in ('from_text', 'from_image', 'from_video'):
            t.add_computed_column(**{f'{col}_duration': pxtf.video.get_duration(t[col])})
        validate_update_status(
            t.insert(prompt='A red dot moving across the screen', image=get_image_files()[0], video=_smallest_video()),
            1,
        )

        row = t.collect()[0]
        assert all(row[f'{col}_duration'] > 0 for col in ('from_text', 'from_image', 'from_video'))

    def test_generate_document(self, uses_db: None) -> None:
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(
            redacted=pxtf.vlmrun.generate_document(
                'Produce a redacted version of this PDF: draw black redaction boxes over all person names. '
                'Return the rendered PDF file, not a text summary.',
                t.doc,
                model=_MODEL,
            )
        )
        # document generation on a large document exceeds the provider's input-token limit
        validate_update_status(t.insert(doc=_smallest_pdf()), 1)

        with open(t.collect()[0]['redacted'], 'rb') as fp:
            assert fp.read(5) == b'%PDF-'
