import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import (
    get_documents,
    get_image_files,
    rerun,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)


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
        """Test chat_completions with an image via file_path auto-upload."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image briefly'},
                    {'type': 'input_file', 'file_path': t.image.localpath},
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
        """Test chat_completions with a PDF document."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What is the title of this document?'},
                    {'type': 'input_file', 'file_path': t.document.localpath},
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

        model = 'vlmrun-orion-1:fast'
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image briefly'},
                    {'type': 'input_file', 'file_path': t.image.localpath},
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
                    {'type': 'input_file', 'file_path': t.document.localpath},
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
        """Test generate_image with file_path returns an edited PIL Image."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import generate_image

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(edited=generate_image('Blur all the faces in this image', file_path=t.image.localpath, timeout=300.0))

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
            annotated=annotate_image('Draw bounding boxes around all objects', file_path=t.image.localpath, timeout=300.0)
        )

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        img = results['annotated'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

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
