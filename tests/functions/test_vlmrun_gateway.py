import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import (
    TESTS_DIR,
    get_audio_file,
    pxt_raises,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

_PDF = str(TESTS_DIR / 'data' / 'documents' / 'Vector_database.pdf')
_IMAGE = str(TESTS_DIR / 'data' / 'images' / 'sewing-threads-smaller.jpg')
_VIDEO = str(TESTS_DIR / 'data' / 'test_videos' / 'test_462_frames.mp4')
_AUDIO = str(TESTS_DIR / 'data' / 'audio' / 'jfk_1961_0109_cityuponahill-excerpt.flac')
_VIDEO_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/tests/data/videos/bangkok_half_res.mp4'
_PDF_URL = 'https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.invoice/sample-invoice.pdf'


class _MockGateway:
    """Records the requests the UDFs send and answers them like the Gateway would."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        # Replies to send, in order, before answering normally: an httpx2.Response, or an exception to raise.
        self.replies: list[Any] = []
        # The text of chat replies; None answers with an empty OCR result.
        self.chat_content: str | None = None
        # Return embeddings in reverse order, or one too few, as a misbehaving server might.
        self.reverse_embeddings = False
        self.drop_embedding = False

    def client(self) -> Any:
        import httpx2
        import openai

        def handle(request: Any) -> Any:
            if request.url.path.endswith('/audio/transcriptions'):
                self.requests.append(
                    {'path': request.url.path, 'content_type': request.headers['content-type'], 'raw': request.content}
                )
                if b'name="response_format"\r\n\r\ntext' in request.content:
                    return httpx2.Response(200, text='hello', headers={'content-type': 'text/plain'})
                return httpx2.Response(200, json={'text': 'hello'})
            body = json.loads(request.content)
            self.requests.append({'path': request.url.path, **body})
            if len(self.replies) > 0:
                reply = self.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply
            if request.url.path.endswith('/embeddings'):
                dim = body.get('dimensions') or {
                    'qwen/qwen3-vl-embedding-2b': 2048,
                    'google/embeddinggemma-2': 768,
                }.get(body['model'], 2)
                # each vector is filled with its input's index, so a test can tell which row got which vector
                data = [
                    {'object': 'embedding', 'index': i, 'embedding': [float(i)] * dim}
                    for i in range(len(body['input']))
                ]
                if self.reverse_embeddings:
                    data.reverse()
                if self.drop_embedding:
                    data.pop()
                return httpx2.Response(200, json={'object': 'list', 'model': body['model'], 'data': data})
            content = self.chat_content
            if content is None:
                content = json.dumps({'model': body['model'], 'content': {'object': 'doc.ocr.lines', 'items': []}})
            return httpx2.Response(
                200,
                json={
                    'id': 'chatcmpl-1',
                    'object': 'chat.completion',
                    'created': 0,
                    'model': body['model'],
                    'choices': [
                        {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}
                    ],
                },
            )

        return openai.AsyncOpenAI(
            api_key='test',
            base_url='https://gateway.vlm.run/v1/openai',
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        )


@pytest.fixture
def gateway(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> _MockGateway:
    """A mock Gateway that the UDFs send their requests to."""
    pytest.importorskip('openai')
    from pixeltable.functions import vlmrun_gateway

    mock = _MockGateway()
    client = mock.client()
    monkeypatch.setattr(vlmrun_gateway, '_client', lambda: client)
    return mock


def _reply(status: int, body: Any = None, *, text: str | None = None, headers: dict[str, str] | None = None) -> Any:
    import httpx2

    if text is not None:
        return httpx2.Response(status, text=text, headers={'content-type': 'text/html', **(headers or {})})
    return httpx2.Response(status, json=body, headers=headers or {})


class TestVlmrunGatewayUnit:
    """Request building and error handling, against a mock Gateway."""

    def test_model_kwargs(self, gateway: _MockGateway) -> None:
        """`model_kwargs` and its `extra_body` reach the request body; the UDF's own arguments win over both."""
        from pixeltable.functions import vlmrun_gateway as gw

        model_kwargs = {'top_k': 20, 'document_dpi': 72, 'extra_body': {'custom': 1, 'document_dpi': 9999}}
        asyncio.run(gw.ocr.py_fns[0](_PDF, model='m', dpi=150, model_kwargs=model_kwargs))
        req = gateway.requests[-1]
        assert req['top_k'] == 20 and req['custom'] == 1 and 'extra_body' not in req
        assert req['document_dpi'] == 150, 'the dpi argument wins'
        assert model_kwargs == {'top_k': 20, 'document_dpi': 72, 'extra_body': {'custom': 1, 'document_dpi': 9999}}

    def test_messages(self, gateway: _MockGateway) -> None:
        from pixeltable.functions import vlmrun_gateway as gw

        img = PIL.Image.new('RGB', (4, 4), 'red')
        content: list[dict[str, Any]] = [
            {'type': 'text', 'text': 'hi'},
            {'type': 'image_url', 'image_url': img},
            {'type': 'image_url', 'image_url': 'https://example.com/a.jpg'},
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/b.jpg', 'detail': 'low'}},
            # the OpenAI form, with a column value inside
            {'type': 'image_url', 'image_url': {'url': img, 'detail': 'high'}},
            {'type': 'document_url', 'document_url': 'Data:application/pdf;base64,JVBERi0='},
            {'type': 'video_url', 'video_url': 'HTTPS://example.com/v.mp4'},
        ]
        messages = [{'role': 'user', 'content': content}]
        asyncio.run(gw.chat_completions.py_fns[0](messages, model='m'))
        parts = gateway.requests[-1]['messages'][0]['content']
        assert parts[0] == {'type': 'text', 'text': 'hi'}
        assert parts[1]['image_url']['url'].startswith('data:image/png;base64,')
        assert parts[2]['image_url'] == {'url': 'https://example.com/a.jpg'}
        assert parts[3]['image_url'] == {'url': 'https://example.com/b.jpg', 'detail': 'low'}
        assert parts[4]['image_url']['url'].startswith('data:image/png;base64,')
        assert parts[4]['image_url']['detail'] == 'high'
        assert parts[5]['document_url'] == {'url': 'Data:application/pdf;base64,JVBERi0='}, 'schemes ignore case'
        assert parts[6]['video_url'] == {'url': 'HTTPS://example.com/v.mp4'}
        assert content[1]['image_url'] is img, 'messages must not be modified'

    def test_message_strings_are_never_read_as_files(self, gateway: _MockGateway, tmp_path: Path) -> None:
        """A string in a media content part could come from any string column, so it is never read from disk."""
        from pixeltable.functions import vlmrun_gateway as gw

        secret = tmp_path / 'secrets.png'
        PIL.Image.new('RGB', (4, 4)).save(secret)
        t = pxt.create_table('test_tbl', {'path': pxt.String})
        t.add_computed_column(
            answer=gw.chat_completions(
                [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': t.path}]}], model='m'
            )
        )
        paths = [str(secret), secret.as_uri(), _PDF, 's3://bucket/clip.mp4', '~/voice.mp3']
        status = t.insert([{'path': p} for p in paths], on_error='ignore')
        assert status.num_excs == len(paths)
        errors = t.select(e=t.answer.errormsg).collect()['e']
        assert all('must be an image column, or an http(s) or data: URL' in e for e in errors), errors
        assert gateway.requests == [], 'nothing was sent'

    def test_documents_must_be_pdfs(self, gateway: _MockGateway, tmp_path: Path) -> None:
        """PDFs are recognized by their contents, as when the row was inserted; other documents fail their own row."""
        from pixeltable.functions import vlmrun_gateway as gw

        notes = tmp_path / 'notes.md'
        notes.write_text('# Notes', encoding='utf-8')
        # a PDF cached from a URL keeps the URL's suffix, such as `.03762` for an arXiv link
        arxiv = tmp_path / 'f3a9c1.03762'
        arxiv.write_bytes(Path(_PDF).read_bytes())
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(pages=gw.ocr(t.doc, model='m'))
        status = t.insert([{'doc': str(notes)}, {'doc': str(arxiv)}], on_error='ignore')
        assert status.num_excs == 1
        errors = [e for e in t.select(e=t.pages.errormsg).collect()['e'] if e is not None]
        assert len(errors) == 1 and "PDF documents only; 'notes.md' is MD" in errors[0], errors
        (req,) = gateway.requests
        assert req['messages'][0]['content'][0]['document_url']['url'].startswith('data:application/pdf;base64,')

    def test_errors(self, gateway: _MockGateway) -> None:
        """Gateway errors keep their message, code and request id, and get the retry behavior the Gateway asks for."""
        import httpx2

        from pixeltable.exec.expr_eval.schedulers import _is_retryable_udf_error
        from pixeltable.functions import vlmrun_gateway as gw

        def run(*replies: Any) -> tuple[pxt.ExternalServiceError, int]:
            gateway.requests.clear()
            gateway.replies[:] = replies
            with pytest.raises(pxt.ExternalServiceError) as info:
                asyncio.run(gw.embeddings.py_fns[0](['hi'], model='m'))
            return info.value, len(gateway.requests)

        def gateway_error(message: str, code: str | None = None) -> dict:
            return {'detail': {'error': {'message': message, 'code': code}}}

        capability = {
            'detail': {
                'error': {
                    'message': "Model 'm' does not support document_url inputs.",
                    'type': 'invalid_request_error',
                    'code': 'capability_violation',
                    'param': 'messages.content.document_url',
                }
            }
        }
        err, calls = run(_reply(400, capability, headers={'x-request-id': 'abc123, abc123'}))
        assert (
            err.error_code == pxt.ErrorCode.PROVIDER_BAD_REQUEST and err.provider_error_code == 'capability_violation'
        )
        assert 'does not support document_url inputs' in str(err) and 'param=messages.content.document_url' in str(err)
        assert '(request id: abc123)' in str(err) and calls == 1

        # The scheduler retries these, honoring Retry-After.
        for reply, error_code, retry_after in [
            (
                _reply(429, {'error': {'message': 'Rate limit exceeded: minute'}}, headers={'retry-after': '12'}),
                pxt.ErrorCode.RATE_LIMITED,
                12.0,
            ),
            (
                _reply(503, gateway_error('Model is busy', 'model_busy'), headers={'retry-after': '30'}),
                pxt.ErrorCode.PROVIDER_OVERLOADED,
                30.0,
            ),
            (_reply(504, gateway_error('No output within 30s', 'ttft_timeout')), pxt.ErrorCode.PROVIDER_ERROR, None),
            (_reply(408, {'error': {'message': 'nope'}}), pxt.ErrorCode.PROVIDER_ERROR, None),
        ]:
            err, calls = run(reply)
            assert err.error_code == error_code and calls == 1, reply.status_code
            assert _is_retryable_udf_error(err) == (True, retry_after), reply.status_code
        assert 'Model is busy' in str(run(_reply(503, gateway_error('Model is busy')))[0])

        # A run that used up the model's whole deadline is not retried.
        err, calls = run(_reply(504, gateway_error('aborted', 'inference_timeout')))
        assert err.error_code == pxt.ErrorCode.PROVIDER_TIMEOUT and calls == 1
        assert _is_retryable_udf_error(err)[0] is False
        assert 'aborted. Try a smaller input, such as fewer PDF pages (`pages` in `ocr()`)' in str(err)

        # The Gateway asks clients to retry 500s and transient 502s; those get one retry here, and the scheduler
        # doesn't retry them again. So does the HTML 502 page of the Gateway's load balancer, which stays out of
        # the message.
        html = '<html><head><title>502 Server Error</title></head><body>temporary error</body></html>'
        for reply in [
            _reply(502, gateway_error('The model failed. Please retry.', 'model_error')),
            _reply(502, gateway_error('The model failed.', 'model_error')),
            _reply(500, {'error': {'message': 'Internal server error', 'code': 'internal_error'}}),
            _reply(502, text=html),
        ]:
            err, calls = run(reply, reply)
            assert calls == 2 and _is_retryable_udf_error(err)[0] is False, reply.status_code
            assert '<html' not in str(err)
        # and a retry that succeeds returns the result
        gateway.requests.clear()
        gateway.replies[:] = [_reply(500, {'error': {'message': 'Internal server error'}})]
        assert len(asyncio.run(gw.embeddings.py_fns[0](['hi'], model='m'))) == 1 and len(gateway.requests) == 2

        # a 502 that isn't transient (such as a document that can't be rasterized) is not retried
        _, calls = run(_reply(502, gateway_error('Lower document_dpi.', 'document_failed')))
        assert calls == 1

        # A plain-string detail (e.g. exhausted credits) is surfaced as-is.
        err, _ = run(_reply(402, {'detail': 'Access denied. Your credits are exhausted.'}))
        assert err.error_code == pxt.ErrorCode.PROVIDER_AUTH_ERROR and 'credits are exhausted' in str(err)

        for status, error_code in [
            (401, pxt.ErrorCode.PROVIDER_AUTH_ERROR),
            (403, pxt.ErrorCode.PROVIDER_AUTH_ERROR),
            (404, pxt.ErrorCode.PROVIDER_BAD_REQUEST),
        ]:
            err, calls = run(_reply(status, {'error': {'message': 'nope'}}))
            assert err.error_code == error_code and calls == 1, status
            assert _is_retryable_udf_error(err)[0] is False, status

        # The client's own timeout is final: the scheduler would otherwise resend a hung request.
        err, calls = run(httpx2.ReadTimeout('timed out'))
        assert err.error_code == pxt.ErrorCode.PROVIDER_TIMEOUT and calls == 1
        assert _is_retryable_udf_error(err)[0] is False

        # A dropped connection is retried by the scheduler.
        err, calls = run(httpx2.ConnectError('Connection reset by peer'))
        assert 'Could not reach the VLM Run Gateway' in str(err) and calls == 1
        assert _is_retryable_udf_error(err)[0] is True

    def test_json_replies(self, gateway: _MockGateway) -> None:
        """A JSON-mode reply that isn't an object fails; a PDF whose pages partly fail is returned with a warning."""
        import logging

        from pixeltable.functions import vlmrun_gateway as gw

        img = PIL.Image.new('RGB', (4, 4))
        gateway.chat_content = '[1, 2]'
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match=r'returned a JSON list instead of an object'):
            asyncio.run(gw.ocr.py_fns[1](img, model='m'))
        gateway.chat_content = 'not json'
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match=r'non-JSON reply'):
            asyncio.run(gw.ocr.py_fns[1](img, model='m'))

        result = {'pages': [{'page_id': 0, 'status': 'ok'}, {'page_id': 1, 'status': 'error'}]}
        gateway.chat_content = json.dumps(result)
        warnings: list[str] = []

        class Recorder(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                warnings.append(record.getMessage())

        handler = Recorder(logging.WARNING)
        logger = logging.getLogger('pixeltable.functions.vlmrun_gateway')
        logger.addHandler(handler)
        try:
            assert asyncio.run(gw.ocr.py_fns[0](_PDF, model='m')) == result
        finally:
            logger.removeHandler(handler)
        assert len(warnings) == 1 and 'page(s) [1]' in warnings[0], warnings

    def test_request_bodies(self, gateway: _MockGateway) -> None:
        """Each UDF sends the content part, method and Gateway fields that the Gateway expects."""
        from pixeltable.functions import vlmrun_gateway as gw

        img = PIL.Image.new('RGB', (4, 4), 'blue')

        result = asyncio.run(gw.ocr.py_fns[0](_PDF, model='m', method='markdown', pages=[[0, 2]], dpi=150))
        assert result['content']['object'] == 'doc.ocr.lines'
        req = gateway.requests[-1]
        assert req['response_format'] == {'type': 'json_object'}
        assert req['method'] == 'markdown' and req['document_pages'] == [[0, 2]] and req['document_dpi'] == 150
        (part,) = req['messages'][0]['content']
        assert part['type'] == 'document_url' and part['document_url']['url'].startswith('data:application/pdf;')

        asyncio.run(gw.ocr.py_fns[1](img, model='m', model_kwargs={'precision': 3}))
        req = gateway.requests[-1]
        assert req['precision'] == 3 and 'method' not in req
        assert req['messages'][0]['content'][0]['type'] == 'image_url'

        for fn, media, part_type in [
            (gw.analyze.py_fns[0], img, 'image_url'),
            (gw.analyze.py_fns[1], _VIDEO, 'video_url'),
        ]:
            asyncio.run(fn(media, model='m', method='segment', method_params={'prompt': 'car'}))
            req = gateway.requests[-1]
            assert req['method'] == 'segment' and req['method_params'] == {'prompt': 'car'}, part_type
            assert req['messages'][0]['content'][0]['type'] == part_type
        assert req['messages'][0]['content'][0]['video_url']['url'].startswith('data:video/mp4;')

        vectors = asyncio.run(gw.embeddings.py_fns[1]([img, img], model='m'))
        assert len(vectors) == 2 and vectors[0].dtype == np.float32
        req = gateway.requests[-1]
        assert req['path'].endswith('/embeddings')
        # the SDK's default is base64; the vectors are parsed as float lists
        assert req['encoding_format'] == 'float'
        # one list of parts per input; a flat list would be pooled into a single vector
        assert [len(x) for x in req['input']] == [1, 1]
        assert req['input'][0][0]['type'] == 'image_url'

        # vectors are matched to inputs by `index`, not by position in the reply
        gateway.reverse_embeddings = True
        vectors = asyncio.run(gw.embeddings.py_fns[0](['a', 'b', 'c'], model='m'))
        assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]
        gateway.reverse_embeddings = False
        gateway.drop_embedding = True
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match=r'returned 2 embeddings for 3 inputs'):
            asyncio.run(gw.embeddings.py_fns[0](['a', 'b', 'c'], model='m'))
        gateway.drop_embedding = False

    def test_images(self, gateway: _MockGateway) -> None:
        """Images are sent as PNG; modes PNG can't store are converted. A CMYK scan used to fail a whole batch."""
        import io

        from pixeltable.functions import vlmrun_gateway as gw

        modes = [('RGB', 'RGB'), ('CMYK', 'RGB'), ('YCbCr', 'RGB'), ('LAB', 'RGB'), ('I', 'RGB'), ('F', 'RGB')]
        modes += [('PA', 'RGBA'), ('RGBa', 'RGBA')]
        asyncio.run(gw.embeddings.py_fns[1]([PIL.Image.new(mode, (8, 6)) for mode, _ in modes], model='m'))
        urls = [x[0]['image_url']['url'] for x in gateway.requests[-1]['input']]
        assert all(url.startswith('data:image/png;base64,') for url in urls)
        sent = [PIL.Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).mode for url in urls]
        assert sent == [expected for _, expected in modes]

    def test_transcription_request(self, gateway: _MockGateway, tmp_path: Path) -> None:
        from pixeltable.functions import vlmrun_gateway as gw

        result = asyncio.run(
            gw.transcriptions.py_fns[0](
                _AUDIO, model='m', model_kwargs={'precision': 'fp16', 'timestamp_granularities': ['word']}
            )
        )
        assert result['text'] == 'hello'
        req = gateway.requests[-1]
        assert req['content_type'].startswith('multipart/form-data')
        assert b'filename="jfk_1961_0109_cityuponahill-excerpt.flac"' in req['raw']
        assert b'Content-Type: audio/flac' in req['raw'] or b'Content-Type: audio/x-flac' in req['raw']
        # Gateway fields go in the form, rather than as SDK arguments the SDK would reject
        assert b'name="precision"\r\n\r\nfp16' in req['raw']
        assert b'name="timestamp_granularities[]"\r\n\r\nword' in req['raw']
        # the plain-text formats come back as a string, which the SDK parses only if it knows the format
        assert asyncio.run(gw.transcriptions.py_fns[0](_AUDIO, model_kwargs={'response_format': 'text'})) == {
            'text': 'hello'
        }

        # a file cached from a URL without an extension is uploaded under a name the Gateway can type
        no_ext = tmp_path / 'download'
        no_ext.write_bytes(Path(_AUDIO).read_bytes())
        asyncio.run(gw.transcriptions.py_fns[0](str(no_ext), model='m'))
        assert b'filename="download.mp3"' in gateway.requests[-1]['raw']

    def test_reserved_model_kwargs(self, gateway: _MockGateway) -> None:
        """Arguments the UDF sets itself are rejected in `model_kwargs`, not silently overridden on the wire."""
        from pixeltable.functions import vlmrun_gateway as gw

        messages = [{'role': 'user', 'content': 'hi'}]
        for call, arg in [
            (gw.chat_completions.py_fns[0](messages, model='m', model_kwargs={'model': 'x'}), 'model'),
            (gw.chat_completions.py_fns[0](messages, model='m', model_kwargs={'messages': []}), 'messages'),
            (gw.embeddings.py_fns[0](['hi'], model='m', model_kwargs={'input': ['x']}), 'input'),
            (gw.transcriptions.py_fns[0](_AUDIO, model_kwargs={'file': 'x'}), 'audio'),
            # model_kwargs['extra_body'] is merged over the request, so it can't set them either
            (gw.chat_completions.py_fns[0](messages, model='m', model_kwargs={'extra_body': {'model': 'x'}}), 'model'),
        ]:
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=rf'pass it as the `{arg}` argument'):
                asyncio.run(call)
        assert gateway.requests == []

    def test_contracts_hold_for_every_udf(self, gateway: _MockGateway) -> None:
        """Rules every UDF must follow, checked on every UDF and overload so that a new one can't skip them."""
        from pixeltable.func.callable_function import CallableFunction
        from pixeltable.functions import vlmrun_gateway as gw

        img = PIL.Image.new('RGB', (4, 4), 'blue')
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'video_url', 'video_url': 'https://example.com/v.mp4'},
                    {'type': 'image_url', 'image_url': img},
                ],
            }
        ]

        chat_keys = ('model', 'messages')
        read_keys = ('model', 'messages', 'response_format')
        embed_keys = ('model', 'input', 'encoding_format')
        # (name, call taking model_kwargs, arguments it sets itself)
        calls: list[tuple[str, Any, tuple[str, ...]]] = [
            (
                'chat_completions',
                lambda kw: gw.chat_completions.py_fns[0](messages, model='m', model_kwargs=kw),
                chat_keys,
            ),
            ('ocr(document)', lambda kw: gw.ocr.py_fns[0](_PDF, model='m', model_kwargs=kw), read_keys),
            ('ocr(image)', lambda kw: gw.ocr.py_fns[1](img, model='m', model_kwargs=kw), read_keys),
            ('analyze(image)', lambda kw: gw.analyze.py_fns[0](img, model='m', model_kwargs=kw), read_keys),
            ('analyze(video)', lambda kw: gw.analyze.py_fns[1](_VIDEO, model='m', model_kwargs=kw), read_keys),
            ('embeddings(text)', lambda kw: gw.embeddings.py_fns[0](['hi'], model='m', model_kwargs=kw), embed_keys),
            ('embeddings(image)', lambda kw: gw.embeddings.py_fns[1]([img], model='m', model_kwargs=kw), embed_keys),
            ('embeddings(video)', lambda kw: gw.embeddings.py_fns[2]([_VIDEO], model='m', model_kwargs=kw), embed_keys),
            ('embeddings(audio)', lambda kw: gw.embeddings.py_fns[3]([_AUDIO], model='m', model_kwargs=kw), embed_keys),
            (
                'embeddings(document)',
                lambda kw: gw.embeddings.py_fns[4]([_PDF], model='m', model_kwargs=kw),
                embed_keys,
            ),
            (
                'transcriptions',
                lambda kw: gw.transcriptions.py_fns[0](_AUDIO, model='m', model_kwargs=kw),
                ('model', 'file'),
            ),
        ]
        udfs = [f for f in gw.__dict__.values() if isinstance(f, CallableFunction)]
        assert len(calls) == sum(len(f.py_fns) for f in udfs), 'a UDF or overload is missing from this test'

        for name, call, own_args in calls:
            gateway.requests.clear()
            # `top_k` is a field the Gateway accepts and the OpenAI SDK doesn't
            model_kwargs = {'precision': 'fp16', 'top_k': 20, 'extra_body': {'custom': 1}}
            asyncio.run(call(model_kwargs))
            assert model_kwargs == {'precision': 'fp16', 'top_k': 20, 'extra_body': {'custom': 1}}, (
                f'{name} modified model_kwargs'
            )
            req = gateway.requests[-1]
            if 'raw' in req:
                assert b'name="precision"\r\n\r\nfp16' in req['raw'], f'{name} dropped a Gateway field'
                assert b'name="top_k"\r\n\r\n20' in req['raw'], f'{name} dropped a field the SDK lacks'
                assert b'name="custom"' in req['raw'], f'{name} dropped extra_body'
            else:
                assert req.get('precision') == 'fp16', f'{name} dropped a Gateway field'
                assert req.get('top_k') == 20, f'{name} dropped a field the SDK lacks'
                assert req.get('custom') == 1, f'{name} dropped extra_body'

            for key in own_args:
                gateway.requests.clear()
                with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=rf'`{key}` cannot be set in `model_kwargs`'):
                    asyncio.run(call({key: 'x'}))
                assert gateway.requests == [], f'{name} sent a request with a conflicting `{key}`'

    def test_tools(self, gateway: _MockGateway) -> None:
        from pixeltable.functions import vlmrun_gateway as gw

        tool = {'name': 'lookup', 'description': 'Look up a value.', 'parameters': {'type': 'object'}}
        messages = [{'role': 'user', 'content': 'hi'}]
        choice = {'auto': False, 'required': False, 'tool': 'lookup', 'parallel_tool_calls': False}
        asyncio.run(gw.chat_completions.py_fns[0](messages, model='m', tools=[tool], tool_choice=choice))
        req = gateway.requests[-1]
        assert req['tools'] == [{'type': 'function', 'function': tool}]
        assert req['tool_choice'] == {'type': 'function', 'function': {'name': 'lookup'}}
        assert req['parallel_tool_calls'] is False
        choice = {'auto': True, 'required': False, 'tool': None, 'parallel_tool_calls': True}
        asyncio.run(gw.chat_completions.py_fns[0](messages, model='m', tools=[tool], tool_choice=choice))
        assert gateway.requests[-1]['tool_choice'] == 'auto' and 'parallel_tool_calls' not in gateway.requests[-1]

        # raw OpenAI tool fields in `model_kwargs` pass through, as in `openai.chat_completions()`; the `tools`
        # argument wins over them
        raw = [{'type': 'function', 'function': {'name': 'raw', 'parameters': {'type': 'object'}}}]
        asyncio.run(gw.chat_completions.py_fns[0](messages, model='m', model_kwargs={'tools': raw}))
        assert gateway.requests[-1]['tools'] == raw
        asyncio.run(gw.chat_completions.py_fns[0](messages, model='m', tools=[tool], model_kwargs={'tools': raw}))
        assert gateway.requests[-1]['tools'] == [{'type': 'function', 'function': tool}]

    def test_table_columns(self, gateway: _MockGateway) -> None:
        """Each overload is picked by column type, return types reach the schema, and null inputs stay null."""
        from pixeltable.functions import vlmrun_gateway as gw

        t = pxt.create_table(
            'test_tbl',
            {
                'text': pxt.String | None,
                'img': pxt.Image | None,
                'doc': pxt.Document | None,
                'video': pxt.Video | None,
                'audio': pxt.Audio | None,
            },
        )
        t.add_computed_column(doc_ocr=gw.ocr(t.doc, model='m'))
        t.add_computed_column(img_ocr=gw.ocr(t.img, model='m'))
        t.add_computed_column(img_regions=gw.analyze(t.img, model='m'))
        t.add_computed_column(video_regions=gw.analyze(t.video, model='m'))
        t.add_computed_column(qwen_vec=gw.embeddings(t.text, model='qwen/qwen3-vl-embedding-2b'))
        t.add_computed_column(text_vec=gw.embeddings(t.text, model='google/embeddinggemma-2'))
        t.add_computed_column(
            small_vec=gw.embeddings(t.text, model='google/embeddinggemma-2', model_kwargs={'dimensions': 256})
        )
        t.add_computed_column(other_vec=gw.embeddings(t.text, model='some/other-model'))
        t.add_computed_column(audio_vec=gw.embeddings(t.audio, model='m', model_kwargs={'dimensions': 4}))
        t.add_computed_column(transcript=gw.transcriptions(t.audio, model='m'))
        t.add_embedding_index(t.img, embedding=gw.embeddings.using(model='qwen/qwen3-vl-embedding-2b'))
        columns = t.get_metadata()['columns']
        for col, shape in [
            ('qwen_vec', '(2048,)'),
            ('text_vec', '(768,)'),
            ('small_vec', '(256,)'),
            ('other_vec', '(None,)'),
            ('audio_vec', '(4,)'),
        ]:
            assert columns[col]['type_'].startswith(f'Array[{shape}'), (col, columns[col]['type_'])

        status = t.insert(
            [{'text': 'hi', 'img': _IMAGE, 'doc': _PDF, 'video': _VIDEO, 'audio': _AUDIO}, {}], on_error='abort'
        )
        assert status.num_excs == 0
        parts = {
            r['messages'][0]['content'][0]['type'] for r in gateway.requests if r['path'].endswith('/chat/completions')
        }
        assert parts == {'document_url', 'image_url', 'video_url'}
        rows = sorted(t.collect(), key=lambda row: row['text'] is None)
        assert rows[0]['text_vec'].shape == (768,) and rows[0]['audio_vec'].shape == (4,)
        assert rows[0]['transcript']['text'] == 'hello'
        cols = ('doc_ocr', 'img_ocr', 'img_regions', 'video_regions', 'text_vec', 'audio_vec', 'transcript')
        assert all(rows[1][c] is None for c in cols), rows[1]
        sim = t.img.similarity(string='a red car')
        assert len(t.order_by(sim, asc=False).limit(1).collect()) == 1

    def test_media_embeddings(self, gateway: _MockGateway, tmp_path: Path) -> None:
        """Media files are embedded one per request; images share one request, like text."""
        from pixeltable.functions import vlmrun_gateway as gw

        vectors = asyncio.run(gw.embeddings.py_fns[3]([_AUDIO, _AUDIO, _AUDIO], model='m'))
        assert len(vectors) == 3
        assert [len(r['input']) for r in gateway.requests] == [1, 1, 1]
        assert gateway.requests[0]['input'][0][0]['type'] == 'audio_url'

        gateway.requests.clear()
        img = PIL.Image.new('RGB', (64, 64), 'green')
        assert len(asyncio.run(gw.embeddings.py_fns[1]([img] * 5, model='m'))) == 5
        assert [len(r['input']) for r in gateway.requests] == [5]

        # a document that isn't a PDF fails the batch before any request is sent (and billed)
        gateway.requests.clear()
        notes = tmp_path / 'notes.md'
        notes.write_text('# Notes', encoding='utf-8')
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'PDF documents only'):
            asyncio.run(gw.embeddings.py_fns[4]([_PDF, str(notes)], model='m'))
        assert gateway.requests == []

    def test_config_sections(self, tmp_path: Path) -> None:
        """`[vlmrun]` and `[vlmrun_gateway]` sections in the config file are accepted and readable."""
        config_file = tmp_path / 'config.toml'
        config_file.write_text(
            '[vlmrun]\napi_key = "shared-key"\n'
            '[vlmrun_gateway]\napi_key = "gw-key"\nbase_url = "https://gw.example.com/v1/openai"\nrate_limit = 160\n',
            encoding='utf-8',
        )
        cmd = (
            'from pixeltable.config import Config\n'
            'c = Config.get()\n'
            'assert c.get_string_value("api_key", section="vlmrun") == "shared-key"\n'
            'assert c.get_string_value("api_key", section="vlmrun_gateway") == "gw-key"\n'
            'assert c.get_string_value("base_url", section="vlmrun_gateway") == "https://gw.example.com/v1/openai"\n'
            'assert c.get_int_value("rate_limit", section="vlmrun_gateway") == 160\n'
        )
        # environment variables take precedence over the config file, so clear them to read the file's values
        env = {k: v for k, v in os.environ.items() if not k.startswith('VLMRUN_')}
        result = subprocess.run(
            (sys.executable, '-c', cmd),
            capture_output=True,
            check=False,
            env={**env, 'PIXELTABLE_CONFIG': str(config_file)},
        )
        assert result.returncode == 0, result.stderr.decode('utf-8')

    def test_client_key_fallback(self, uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Gateway client uses its own key if set, else the `vlmrun` key, else reports a missing key."""
        pytest.importorskip('openai')
        from pixeltable.config import Config
        from pixeltable.env import _client_factories

        init = _client_factories['vlmrun_gateway'].init_fn
        for var in ('VLMRUN_API_KEY', 'VLMRUN_GATEWAY_API_KEY', 'VLMRUN_GATEWAY_BASE_URL'):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv('VLMRUN_API_KEY', 'shared-key')
        client = init()
        assert client.api_key == 'shared-key'
        assert str(client.base_url) == 'https://gateway.vlm.run/v1/openai/'
        assert client.max_retries == 0

        client = init(api_key='gw-key', base_url='https://gw.example.com/v1/openai')
        assert client.api_key == 'gw-key'
        assert str(client.base_url) == 'https://gw.example.com/v1/openai/'

        # neither key configured (the developer's config file may hold one, so hide it)
        monkeypatch.delenv('VLMRUN_API_KEY')
        get_string_value = Config.get().get_string_value
        monkeypatch.setattr(
            Config.get(),
            'get_string_value',
            lambda key, section='pixeltable': None if section == 'vlmrun' else get_string_value(key, section),
        )
        with pxt_raises(pxt.ErrorCode.MISSING_CREDENTIALS, match=r'no VLM Run API key is configured'):
            init()


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestVlmrunGateway:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import chat_completions

        t = pxt.create_table('test_tbl', {'img': pxt.Image, 'video': pxt.Video})
        image_messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image in one sentence.'},
                    {'type': 'image_url', 'image_url': t.img},
                ],
            }
        ]
        video_messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this video in one sentence.'},
                    # the Gateway fetches the URL; a local video column can't be read from `messages`
                    {'type': 'video_url', 'video_url': t.video.fileurl},
                ],
            }
        ]
        # 27b reasons before it answers; 60 tokens ended mid-reasoning with empty content (checked live)
        t.add_computed_column(
            image_answer=chat_completions(
                image_messages, model='qwen/qwen3.8-27b', model_kwargs={'max_tokens': 400, 'top_k': 20}
            )
        )
        t.add_computed_column(
            video_answer=chat_completions(
                video_messages,
                model='qwen/qwen3.8-27b',
                model_kwargs={'max_tokens': 400, 'video_fps': 1.0, 'video_max_frames': 4},
            )
        )
        t.add_computed_column(
            fields=chat_completions(
                image_messages,
                model='qwen/qwen3.8-27b',
                model_kwargs={
                    'response_format': {
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'colors',
                            'schema': {
                                'type': 'object',
                                'properties': {'main_color': {'type': 'string'}},
                                'required': ['main_color'],
                            },
                        },
                    }
                },
            )
        )
        validate_update_status(t.insert(img=_IMAGE, video=_VIDEO_URL), 1)
        row = t.collect()[0]
        image_answer = row['image_answer']['choices'][0]['message']['content'].lower()
        assert any(w in image_answer for w in ('thread', 'spool', 'yarn', 'sewing')), image_answer
        video_answer = row['video_answer']['choices'][0]['message']['content'].lower()
        assert any(w in video_answer for w in ('street', 'city', 'traffic', 'road', 'urban')), video_answer
        for col in ('image_answer', 'video_answer'):
            assert row[col]['usage']['cost'] >= 0, col
        import json

        assert isinstance(json.loads(row['fields']['choices'][0]['message']['content'])['main_color'], str)

    def test_ocr(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import ocr

        t = pxt.create_table('test_tbl', {'doc': pxt.Document, 'img': pxt.Image})
        t.add_computed_column(pages=ocr(t.doc, pages=[0], dpi=96))
        t.add_computed_column(markdown=ocr(t.img))
        t.add_computed_column(lines=ocr(t.img, model='paddleocr/pp-ocrv6', method='ocr'))
        validate_update_status(t.insert(doc=_PDF, img=_IMAGE), 1)
        row = t.collect()[0]

        pages = row['pages']
        assert pages['document_npages'] == 1
        assert pages['pages'][0]['content']['object'] == 'doc.page.blocks'
        assert 'vector' in pages['pages'][0]['content']['items'][0]['text'].lower()
        assert isinstance(row['markdown']['content'], str)
        assert row['lines']['content']['object'] == 'doc.ocr.lines'
        assert len(row['lines']['content']['items']) > 0, 'no text lines found'
        for item in row['lines']['content']['items']:
            assert len(item['bbox_xywh']) == 4

    def test_analyze(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import analyze

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_computed_column(
            detections=analyze(
                t.img,
                model='facebook/sam3.1',
                method='segment',
                method_params={'prompt': 'thread', 'mask_format': 'none', 'polygons': False},
            )
        )
        validate_update_status(t.insert(img=_IMAGE), 1)
        result = t.collect()[0]['detections']
        assert result['content']['object'] == 'img.segment.masks'
        assert len(result['content']['items']) > 0, 'no threads segmented'
        assert result['image_width'] > 0

        v = pxt.create_table('test_video_tbl', {'video': pxt.Video})
        v.add_computed_column(
            poses=analyze(
                v.video,
                model='usyd-community/vitpose-plus-large',
                model_kwargs={'video_fps': 1.0, 'video_max_frames': 2},
            )
        )
        validate_update_status(v.insert(video=_VIDEO), 1)
        result = v.collect()[0]['poses']
        assert result['content']['object'] == 'vid.pose.kpts'
        assert result['video_nframes'] > 0

    def test_embeddings(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import embeddings

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_embedding_index(t.img, embedding=embeddings.using(model='qwen/qwen3-vl-embedding-2b'))
        red = PIL.Image.new('RGB', (64, 64), (220, 20, 20))
        blue = PIL.Image.new('RGB', (64, 64), (20, 20, 220))
        validate_update_status(t.insert([{'img': red}, {'img': blue}, {'img': _IMAGE}]), 3)
        sim = t.img.similarity(string='a plain solid red square')
        top = t.order_by(sim, asc=False).limit(1).select(t.img).collect()[0]['img']
        assert top.getpixel((0, 0)) == (220, 20, 20)

        t.add_computed_column(text_vec=embeddings('hello', model='google/embeddinggemma-2'))
        t.add_computed_column(
            small_vec=embeddings('hello', model='google/embeddinggemma-2', model_kwargs={'dimensions': 256})
        )
        row = t.limit(1).collect()[0]
        assert row['text_vec'].shape == (768,)
        assert row['small_vec'].shape == (256,)

        m = pxt.create_table('test_media_tbl', {'doc': pxt.Document, 'audio': pxt.Audio, 'video': pxt.Video})
        model_kwargs = {'document_pages': [0]}
        m.add_computed_column(doc_vec=embeddings(m.doc, model='google/embeddinggemma-2', model_kwargs=model_kwargs))
        m.add_computed_column(audio_vec=embeddings(m.audio, model='google/embeddinggemma-2'))
        m.add_computed_column(video_vec=embeddings(m.video, model='google/embeddinggemma-2'))
        audio = get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac')
        validate_update_status(m.insert(doc=_PDF, audio=audio, video=_VIDEO), 1)
        row = m.collect()[0]
        for col in ('doc_vec', 'audio_vec', 'video_vec'):
            assert row[col].shape == (768,), col
        assert not np.allclose(row['doc_vec'], row['audio_vec']), 'different media must embed differently'
        assert not np.allclose(row['audio_vec'], row['video_vec']), 'different media must embed differently'

    def test_transcriptions(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import transcriptions

        t = pxt.create_table('test_tbl', {'audio': pxt.Audio})
        t.add_computed_column(transcript=transcriptions(t.audio))
        validate_update_status(t.insert(audio=get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac')), 1)
        result = t.collect()[0]['transcript']
        assert 'city upon a hill' in result['text'].lower()
        assert result['usage']['cost'] >= 0
        assert result['duration'] > 0, 'the json format reports the audio duration'

    def test_errors(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import chat_completions

        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        messages = [{'role': 'user', 'content': [{'type': 'document_url', 'document_url': t.doc.fileurl}]}]
        t.add_computed_column(answer=chat_completions(messages, model='qwen/qwen3.5-0.8b'))
        status = t.insert(doc=_PDF_URL, on_error='ignore')
        assert status.num_excs == 1
        errormsg = t.select(msg=t.answer.errormsg).collect()[0]['msg']
        assert 'capability_violation' in errormsg
        assert 'does not support document_url' in errormsg
