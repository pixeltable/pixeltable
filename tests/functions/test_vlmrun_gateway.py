import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from tests.functions.tool_utils import run_tool_invocations_test

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
        assert isinstance(json.loads(row['fields']['choices'][0]['message']['content'])['main_color'], str)

    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.openai import invoke_tools
        from pixeltable.functions.vlmrun_gateway import chat_completions

        def make_table(tools: pxt.func.Tools, tool_choice: pxt.func.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String | None}, if_exists='replace')
            t.add_computed_column(
                response=chat_completions(
                    messages=[{'role': 'user', 'content': t.prompt}],
                    model='qwen/qwen3.8-27b',
                    tools=tools,
                    tool_choice=tool_choice,
                )
            )
            t.add_computed_column(tool_calls=invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table)

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
        t.add_computed_column(answer=chat_completions(messages, model='facebook/sam3.1'))
        status = t.insert(doc=_PDF_URL, on_error='ignore')
        assert status.num_excs == 1
        errormsg = t.select(msg=t.answer.errormsg).collect()[0]['msg']
        assert 'capability_violation' in errormsg
        assert 'does not support document_url' in errormsg

    def test_message_strings_are_never_read_as_files(self, uses_db: None, tmp_path: Path) -> None:
        """A string in a media content part could come from any string column, so it is never read from disk."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import chat_completions

        secret = tmp_path / 'secrets.png'
        PIL.Image.new('RGB', (4, 4)).save(secret)
        t = pxt.create_table('test_tbl', {'path': pxt.String})
        t.add_computed_column(
            answer=chat_completions(
                [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': t.path}]}], model='qwen/qwen3.5-0.8b'
            )
        )
        paths = [str(secret), secret.as_uri(), _PDF, 's3://bucket/clip.mp4', '~/voice.mp3']
        status = t.insert([{'path': p} for p in paths], on_error='ignore')
        assert status.num_excs == len(paths)
        errors = t.select(e=t.answer.errormsg).collect()['e']
        assert all('must be an image column, or an http(s) or data: URL' in e for e in errors), errors

    def test_documents_must_be_pdfs(self, uses_db: None, tmp_path: Path) -> None:
        """PDFs are recognized by their contents, as when the row was inserted; other documents fail their own row."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import ocr

        notes = tmp_path / 'notes.md'
        notes.write_text('# Notes', encoding='utf-8')
        # a PDF cached from a URL keeps the URL's suffix, such as `.03762` for an arXiv link
        arxiv = tmp_path / 'f3a9c1.03762'
        arxiv.write_bytes(Path(_PDF).read_bytes())
        t = pxt.create_table('test_tbl', {'doc': pxt.Document})
        t.add_computed_column(pages=ocr(t.doc, pages=[0], dpi=96))
        status = t.insert([{'doc': str(notes)}, {'doc': str(arxiv)}], on_error='ignore')
        assert status.num_excs == 1
        rows = t.select(pages=t.pages, e=t.pages.errormsg).collect()
        errors = [e for e in rows['e'] if e is not None]
        assert len(errors) == 1 and "PDF documents only; 'notes.md' is MD" in errors[0], errors
        (pages,) = [p for p in rows['pages'] if p is not None]
        assert pages['document_npages'] == 1

    def test_model_kwargs(self, uses_db: None) -> None:
        """`model_kwargs` is not modified, and arguments a UDF sets itself are rejected there for every UDF."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.func.callable_function import CallableFunction
        from pixeltable.functions import vlmrun_gateway as gw

        img = PIL.Image.new('RGB', (4, 4), 'blue')
        model_kwargs = {'document_dpi': 72, 'extra_body': {'document_pages': [0]}}
        result = asyncio.run(gw.ocr.py_fns[0](_PDF, dpi=96, model_kwargs=model_kwargs))
        assert result['document_npages'] == 1, 'extra_body reached the Gateway'
        assert model_kwargs == {'document_dpi': 72, 'extra_body': {'document_pages': [0]}}

        messages = [{'role': 'user', 'content': 'hi'}]
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
            for key in own_args:
                # extra_body is merged over the request, so it can't set them either
                for kw in ({key: 'x'}, {'extra_body': {key: 'x'}}):
                    with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=rf'`{key}` cannot be set in `model_kwargs`'):
                        asyncio.run(call(kw))
                    assert kw in ({key: 'x'}, {'extra_body': {key: 'x'}}), f'{name} modified model_kwargs'

    def test_image_modes(self, uses_db: None) -> None:
        """Images in modes PNG can't store are converted before sending. A CMYK scan used to fail a whole batch."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
        from pixeltable.functions.vlmrun_gateway import embeddings

        modes = ['RGB', 'CMYK', 'YCbCr', 'LAB', 'I', 'F', 'PA', 'RGBa']
        vectors = asyncio.run(
            embeddings.py_fns[1]([PIL.Image.new(mode, (8, 6)) for mode in modes], model='qwen/qwen3-vl-embedding-2b')
        )
        assert [v.shape for v in vectors] == [(2048,)] * len(modes)

    def test_null_inputs_and_schema(self, uses_db: None) -> None:
        """Return types reach the schema, and null inputs stay null without a request."""
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('vlmrun_gateway')
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
        t.add_computed_column(doc_ocr=gw.ocr(t.doc))
        t.add_computed_column(img_ocr=gw.ocr(t.img))
        t.add_computed_column(img_regions=gw.analyze(t.img, model='facebook/sam3.1'))
        t.add_computed_column(video_regions=gw.analyze(t.video, model='usyd-community/vitpose-plus-large'))
        t.add_computed_column(qwen_vec=gw.embeddings(t.text, model='qwen/qwen3-vl-embedding-2b'))
        t.add_computed_column(text_vec=gw.embeddings(t.text, model='google/embeddinggemma-2'))
        t.add_computed_column(
            small_vec=gw.embeddings(t.text, model='google/embeddinggemma-2', model_kwargs={'dimensions': 256})
        )
        t.add_computed_column(other_vec=gw.embeddings(t.text, model='some/other-model'))
        t.add_computed_column(transcript=gw.transcriptions(t.audio))
        columns = t.get_metadata()['columns']
        for col, shape in [
            ('qwen_vec', '(2048,)'),
            ('text_vec', '(768,)'),
            ('small_vec', '(256,)'),
            ('other_vec', '(None,)'),
        ]:
            assert columns[col]['type_'].startswith(f'Array[{shape}'), (col, columns[col]['type_'])

        validate_update_status(t.insert([{}], on_error='abort'), 1)
        row = t.collect()[0]
        cols = ('doc_ocr', 'img_ocr', 'img_regions', 'video_regions', 'text_vec', 'other_vec', 'transcript')
        assert all(row[c] is None for c in cols), row
