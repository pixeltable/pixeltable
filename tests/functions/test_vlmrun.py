import asyncio
import os

import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import (
    get_documents,
    get_image_files,
    pxt_raises,
    rerun,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.local('UDF/integration test')


class TestVLMRunMessageResolution:
    """Credential-free unit tests for media resolution in `chat_completions` messages."""

    @pytest.fixture
    def patched_uploads(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        from pixeltable.functions import vlmrun as vlmrun_mod

        uploads: list[str] = []

        async def fake_upload_file(file_path: object) -> str:
            await asyncio.sleep(0)
            uploads.append(f'file:{file_path}')
            return f'file_{len(uploads)}'

        async def fake_upload_image(image: PIL.Image.Image) -> str:
            await asyncio.sleep(0)
            uploads.append('image')
            return f'file_{len(uploads)}'

        monkeypatch.setattr(vlmrun_mod, '_upload_file', fake_upload_file)
        monkeypatch.setattr(vlmrun_mod, '_upload_image', fake_upload_image)
        return uploads

    def test_pil_image_uploaded(self, patched_uploads: list[str]) -> None:
        from pixeltable.functions.vlmrun import _resolve_messages

        img = PIL.Image.new('RGB', (4, 4))
        content: list[dict] = [{'type': 'image_url', 'image_url': img}]
        messages = [{'role': 'user', 'content': content}]
        resolved = asyncio.run(_resolve_messages(messages))
        item = resolved[0]['content'][0]
        assert item == {'type': 'input_file', 'file_id': 'file_1'}
        assert patched_uploads == ['image']
        # original messages are not mutated
        assert isinstance(content[0]['image_url'], PIL.Image.Image)

    def test_local_paths_uploaded(self, patched_uploads: list[str]) -> None:
        from pixeltable.functions.vlmrun import _resolve_messages

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'file_url', 'file_url': '/tmp/doc.pdf'},
                    {'type': 'video_url', 'video_url': '/tmp/clip.mp4'},
                ],
            }
        ]
        resolved = asyncio.run(_resolve_messages(messages))
        assert resolved[0]['content'][0] == {'type': 'input_file', 'file_id': 'file_1'}
        assert resolved[0]['content'][1] == {'type': 'input_file', 'file_id': 'file_2'}
        assert patched_uploads == ['file:/tmp/doc.pdf', 'file:/tmp/clip.mp4']

    def test_urls_passed_through(self, patched_uploads: list[str]) -> None:
        from pixeltable.functions.vlmrun import _resolve_messages

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': 'https://example.com/a.jpg'},
                    {'type': 'image_url', 'image_url': 'data:image/png;base64,AAAA'},
                    {'type': 'video_url', 'video_url': {'url': 'https://example.com/v.mp4'}},
                ],
            }
        ]
        resolved = asyncio.run(_resolve_messages(messages))
        content = resolved[0]['content']
        assert content[0] == {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.jpg'}}
        assert content[1] == {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}
        assert content[2] == {'type': 'video_url', 'video_url': {'url': 'https://example.com/v.mp4'}}
        assert patched_uploads == []

    def test_input_file_handling(self, patched_uploads: list[str]) -> None:
        from pixeltable.functions.vlmrun import _resolve_messages

        # existing file_id passes through; file_path is uploaded
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_file', 'file_id': 'file_abc'},
                    {'type': 'input_file', 'file_path': '/tmp/doc.pdf'},
                ],
            }
        ]
        resolved = asyncio.run(_resolve_messages(messages))
        assert resolved[0]['content'][0] == {'type': 'input_file', 'file_id': 'file_abc'}
        assert resolved[0]['content'][1] == {'type': 'input_file', 'file_id': 'file_1'}

        # neither file_id nor file_path is an error
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='file_id'):
            asyncio.run(_resolve_messages([{'role': 'user', 'content': [{'type': 'input_file'}]}]))

    def test_string_content_passed_through(self, patched_uploads: list[str]) -> None:
        from pixeltable.functions.vlmrun import _resolve_messages

        messages = [{'role': 'user', 'content': 'plain text'}]
        assert asyncio.run(_resolve_messages(messages)) == messages

    def test_unsupported_extension_rejected(self) -> None:
        from pixeltable.functions.vlmrun import _upload_file

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='Unsupported file format'):
            asyncio.run(_upload_file('/tmp/audio.mp3'))

    def test_classify_media_ref(self) -> None:
        from pixeltable.functions.vlmrun import _classify_media_ref

        assert _classify_media_ref('vid_a1b2c3') == ('artifact', 'vid_a1b2c3')
        assert _classify_media_ref('vid_4d0e56.mp4') == ('artifact', 'vid_4d0e56')
        assert _classify_media_ref('doc_9f8e7d.pdf') == ('artifact', 'doc_9f8e7d')
        assert _classify_media_ref('url_f41446') == ('artifact', 'url_f41446')
        assert _classify_media_ref(' https://cdn.example.com/v.mp4 ') == ('url', 'https://cdn.example.com/v.mp4')
        with pxt_raises(pxt.ErrorCode.PROVIDER_ERROR, match='unrecognized media reference'):
            _classify_media_ref('not-a-ref.mp4')


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestVLMRun:
    def test_chat_completions_text(self, uses_db: None) -> None:
        """Test chat_completions with a text-only message."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        messages = [{'role': 'user', 'content': t.prompt}]
        t.add_computed_column(response=chat_completions(messages))
        t.add_computed_column(answer=t.response['choices'][0]['message']['content'])

        validate_update_status(t.insert(prompt='What is 2+2?'), 1)
        results = t.collect()

        response = results['response'][0]
        assert 'choices' in response
        assert len(response['choices']) > 0
        assert 'session_id' not in response

        answer = results['answer'][0]
        assert answer is not None
        assert len(answer) > 0

    def test_chat_completions_image(self, uses_db: None) -> None:
        """Test chat_completions with an image column referenced in the messages."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image briefly'},
                    {'type': 'image_url', 'image_url': t.image},
                ],
            }
        ]
        t.add_computed_column(response=chat_completions(messages))
        t.add_computed_column(description=t.response['choices'][0]['message']['content'])

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        response = results['response'][0]
        assert 'choices' in response
        assert len(response['choices']) > 0
        assert 'session_id' not in response

        description = results['description'][0]
        assert description is not None
        assert len(description) > 0

    def test_chat_completions_document(self, uses_db: None) -> None:
        """Test chat_completions with a document column referenced in the messages."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What is the title of this document?'},
                    {'type': 'file_url', 'file_url': t.document},
                ],
            }
        ]
        t.add_computed_column(response=chat_completions(messages))
        t.add_computed_column(title=t.response['choices'][0]['message']['content'])

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'

        validate_update_status(t.insert(document=pdf_files[0]), 1)
        results = t.collect()

        title = results['title'][0]
        assert title is not None
        assert len(title) > 0

    def test_chat_completions_model_tier(self, uses_db: None) -> None:
        """Test chat_completions with a non-default model tier."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        model = 'vlmrun-orion-2:fast'
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image briefly'},
                    {'type': 'image_url', 'image_url': t.image},
                ],
            }
        ]
        t.add_computed_column(response=chat_completions(messages, model=model))
        t.add_computed_column(description=t.response['choices'][0]['message']['content'])

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        response = results['response'][0]
        assert response['model'] == model

        description = results['description'][0]
        assert len(description) > 0

    def test_chat_completions_toolset(self, uses_db: None) -> None:
        """Test chat_completions with the toolset parameter via model_kwargs."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Extract all text from this document'},
                    {'type': 'file_url', 'file_url': t.document},
                ],
            }
        ]
        t.add_computed_column(
            response=chat_completions(messages, model_kwargs={'extra_body': {'toolsets': ['document']}})
        )
        t.add_computed_column(extracted=t.response['choices'][0]['message']['content'])

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'

        validate_update_status(t.insert(document=pdf_files[0]), 1)
        results = t.collect()

        response = results['response'][0]
        assert 'choices' in response
        assert len(response['choices']) > 0

        extracted = results['extracted'][0]
        assert extracted is not None
        assert len(extracted) > 0

    @pytest.mark.expensive
    def test_generate_image(self, uses_db: None) -> None:
        """Test generate_image returns a PIL Image (text-to-image)."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import generate_image

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate_image(t.prompt, timeout=300.0))

        validate_update_status(t.insert(prompt='A red circle on a white background'), 1)
        results = t.collect()

        img = results['image'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

    @pytest.mark.expensive
    def test_generate_image_edit(self, uses_db: None) -> None:
        """Test generate_image with an input image column returns an edited PIL Image."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import generate_image

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(edited=generate_image('Blur all the faces in this image', image=t.image, timeout=300.0))

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        img = results['edited'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

    @pytest.mark.expensive
    def test_annotate_image(self, uses_db: None) -> None:
        """Test annotate_image returns a PIL Image with viz annotations."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import annotate_image

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(
            annotated=annotate_image('Draw bounding boxes around all objects', t.image, timeout=300.0)
        )

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        img = results['annotated'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_batch_error_isolation(self, uses_db: None) -> None:
        """Test that a multi-row insert runs all rows and a failing row doesn't affect the others."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'path': pxt.String})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image in one short sentence.'},
                    {'type': 'image_url', 'image_url': t.path},
                ],
            }
        ]
        t.add_computed_column(response=chat_completions(messages, model='vlmrun-orion-2:fast'))

        image_file = get_image_files()[0]
        txt_files = sorted(f for f in get_documents() if f.endswith('.txt'))
        assert len(txt_files) > 0
        # one valid row, one row that fails media resolution (.txt is unsupported)
        status = t.insert([{'path': image_file}, {'path': txt_files[0]}], on_error='ignore')
        assert status.num_rows == 2
        assert status.num_excs >= 1

        res = t.select(t.path, t.response, err=t.response.errormsg).collect()
        by_path = {r['path']: r for r in res}
        good = by_path[image_file]
        bad = by_path[txt_files[0]]
        assert good['response'] is not None and len(good['response']['choices']) > 0
        assert bad['response'] is None
        assert 'Unsupported file format' in str(bad['err'])

    @pytest.mark.very_expensive
    @rerun(reruns=1, reruns_delay=8)
    def test_generate_document(self, uses_db: None) -> None:
        """Test generate_document returns a transformed PDF (redaction).

        Document generation is the provider's slowest operation (an agentic render/detect/redact/
        re-render pipeline); wall-clock is high-variance, from ~2 to 20+ minutes under load.
        """
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import generate_document

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(
            redacted=generate_document(
                'Produce a redacted version of this PDF: draw black redaction boxes over all person names. '
                'Return the rendered PDF file, not a text summary.',
                t.document,
                timeout=900.0,
            )
        )

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'
        # use the smallest PDF: document generation on large documents exceeds provider input-token limits
        smallest_pdf = min(pdf_files, key=os.path.getsize)

        validate_update_status(t.insert(document=smallest_pdf), 1)
        results = t.collect()

        redacted_path = results['redacted'][0]
        assert redacted_path is not None
        with open(redacted_path, 'rb') as fp:
            assert fp.read(5) == b'%PDF-', 'expected a rendered PDF document'

    @pytest.mark.expensive
    def test_generate_video(self, uses_db: None) -> None:
        """Test generate_video returns a video file path."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import generate_video

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(video=generate_video(t.prompt, timeout=600.0))

        validate_update_status(t.insert(prompt='A red dot'), 1)
        results = t.collect()

        video_path = results['video'][0]
        assert video_path is not None
        assert isinstance(video_path, str)
        assert len(video_path) > 0
