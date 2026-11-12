import json
import os
from pathlib import Path
from typing import Any, Callable

import httpx
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.config import Config
from pixeltable.functions.vlmrun import _require_https
from pixeltable.runtime import get_runtime, reset_runtime

from ..utils import (
    get_documents,
    get_image_files,
    get_video_files,
    pxt_raises,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

_MODEL = 'vlmrun-orion-2:fast'
# video and document generation on the fast tier returned made-up artifact ids in live tests
_GENERATION_MODEL = 'vlmrun-orion-2:auto'


def _smallest_pdf() -> str:
    return min((d for d in get_documents() if d.endswith('.pdf')), key=os.path.getsize)


def _smallest_video() -> str:
    return min((v for v in get_video_files() if v.endswith('.mp4')), key=os.path.getsize)


class TestVLMRunClient:
    def test_client_closed_on_reset(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('vlmrun')
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

    @pytest.mark.parametrize(
        ('udf', 'key'),
        [
            ('generate_image', 'toolsets'),
            ('generate_image', 'inputs'),
            ('generate_image', 'batch'),
            ('chat_completions', 'model'),
        ],
    )
    def test_reserved_model_kwargs(self, uses_db: None, udf: str, key: str) -> None:
        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        model_kwargs = {key: 'x'}
        if udf == 'generate_image':
            t.add_computed_column(out=pxtf.vlmrun.generate_image(t.prompt, model=_MODEL, model_kwargs=model_kwargs))
        else:
            messages = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(out=pxtf.vlmrun.chat_completions(messages, model=_MODEL, model_kwargs=model_kwargs))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=f'cannot contain `{key}`'):
            t.insert(prompt='A red circle')

    def test_generate_document_submit_poll(self, uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {'execute': 0, 'artifacts': 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if request.method == 'POST' and path.endswith('/agent/execute'):
                calls['execute'] += 1
                body = json.loads(request.content)
                assert body['batch'] is True
                assert body['toolsets'] == ['document']
                return httpx.Response(
                    200, json={'id': 'exec_test', 'status': 'completed', 'response': {'document': {'id': 'doc_abcdef'}}}
                )
            if request.method == 'GET' and path.endswith('/artifacts'):
                calls['artifacts'] += 1
                assert request.url.params['object_id'] == 'doc_abcdef'
                assert request.url.params['execution_id'] == 'exec_test'
                return httpx.Response(200, content=b'%PDF-1.4 mock\n')
            raise AssertionError(f'unexpected {request.method} {path}')

        _install_mock_vlmrun(monkeypatch, handler)
        try:
            t = pxt.create_table('test_tbl', {'doc': pxt.Document})
            t.add_computed_column(
                redacted=pxtf.vlmrun.generate_document('Redact names', t.doc, model=_GENERATION_MODEL)
            )
            validate_update_status(t.insert(doc=_smallest_pdf()), 1)
            with open(t.collect()[0]['redacted'], 'rb') as fp:
                assert fp.read(5) == b'%PDF-'
            assert calls == {'execute': 1, 'artifacts': 1}
        finally:
            reset_runtime()

    def test_generate_disconnect_after_accept_is_not_retried(
        self, uses_db: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        posts = {'n': 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == 'POST' and request.url.path.endswith('/agent/execute'):
                posts['n'] += 1
                raise httpx.ReadError('', request=request)
            raise AssertionError(f'unexpected {request.method} {request.url.path}')

        _install_mock_vlmrun(monkeypatch, handler)
        try:
            t = pxt.create_table('test_tbl', {'doc': pxt.Document})
            t.add_computed_column(
                redacted=pxtf.vlmrun.generate_document('Redact names', t.doc, model=_GENERATION_MODEL)
            )
            status = t.insert(doc=_smallest_pdf(), on_error='ignore')
            assert status.num_excs == 1
            assert posts['n'] == 1
        finally:
            reset_runtime()


def _install_mock_vlmrun(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    skip_test_if_not_installed('vlmrun')
    monkeypatch.setenv('VLMRUN_API_KEY', 'test-key')
    reset_runtime()
    client = get_runtime().get_client('vlmrun')
    client._http = httpx.AsyncClient(
        base_url=client._http.base_url,
        timeout=httpx.Timeout(None, connect=30.0),
        transport=httpx.MockTransport(handler),
        event_hooks={'request': [_require_https]},
    )

    async def fake_upload(_file: object) -> str:  # noqa: RUF029
        return 'file_test'

    async def fake_delete(_file_ids: list[str]) -> None:  # noqa: RUF029
        return None

    monkeypatch.setattr(client, 'upload', fake_upload)
    monkeypatch.setattr(client, 'delete_files', fake_delete)


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestVLMRun:
    @pytest.mark.parametrize(
        ('col_type', 'value', 'match'),
        [
            (
                pxt.Document,
                lambda: next(d for d in get_documents() if d.endswith('.txt')),
                r'Unsupported file format: \.txt',
            ),
            (pxt.String, lambda: '/no/such/dir/file.pdf', 'Not a file: /no/such/dir/file.pdf'),
        ],
    )
    def test_bad_local_file(self, uses_db: None, col_type: Any, value: Callable[[], str], match: str) -> None:
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'file': col_type})
        messages = [{'role': 'user', 'content': [{'type': 'file_url', 'file_url': t.file}]}]
        t.add_computed_column(response=pxtf.vlmrun.chat_completions(messages, model=_MODEL))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=match):
            t.insert(file=value())

    @pytest.mark.parametrize(
        ('col_type', 'value', 'match'),
        [(pxt.Int, 1, '`image_url` must be a media column'), (pxt.Image | None, None, '`image_url` is null')],
    )
    def test_invalid_media_value(self, uses_db: None, col_type: Any, value: Any, match: str) -> None:
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'media': col_type})
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': t.media}]}]
        t.add_computed_column(response=pxtf.vlmrun.chat_completions(messages, model=_MODEL))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=match):
            t.insert(media=value)

    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('vlmrun')
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
        skip_test_if_not_installed('vlmrun')
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
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'image': pxt.Image, 'video': pxt.Video})
        t.add_computed_column(from_text=pxtf.vlmrun.generate_video(t.prompt, model=_GENERATION_MODEL))
        t.add_computed_column(from_image=pxtf.vlmrun.generate_video(t.prompt, image=t.image, model=_GENERATION_MODEL))
        t.add_computed_column(from_video=pxtf.vlmrun.generate_video(t.prompt, video=t.video, model=_GENERATION_MODEL))
        for col in ('from_text', 'from_image', 'from_video'):
            t.add_computed_column(**{f'{col}_duration': pxtf.video.get_duration(t[col])})
        validate_update_status(
            t.insert(prompt='A red dot moving across the screen', image=get_image_files()[0], video=_smallest_video()),
            1,
        )

        row = t.collect()[0]
        assert all(row[f'{col}_duration'] > 0 for col in ('from_text', 'from_image', 'from_video'))

    def test_generate_document(self, uses_db: None) -> None:
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(
            redacted=pxtf.vlmrun.generate_document(
                'Produce a redacted version of this PDF: draw black redaction boxes over all person names. '
                'Return the rendered PDF file, not a text summary.',
                t.doc,
                model=_GENERATION_MODEL,
            )
        )
        # document generation on a large document exceeds the provider's input-token limit
        validate_update_status(t.insert(doc=_smallest_pdf()), 1)

        with open(t.collect()[0]['redacted'], 'rb') as fp:
            assert fp.read(5) == b'%PDF-'
