import pytest

import pixeltable as pxt

from ..utils import (
    get_documents,
    get_image_files,
    get_video_files,
    rerun,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestVLMRun:
    def test_upload_file_image(self, reset_db: None) -> None:
        """Test upload_file with an image."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import upload_file

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_file(t.image.localpath))

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        file_id = results['file_id'][0]
        assert file_id is not None
        assert len(file_id) > 0

        print(f"\n{'=' * 70}")
        print("upload_file (Image) Test Results")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"File ID: {file_id}")
        print(f"{'=' * 70}\n")

    def test_upload_file_pdf(self, reset_db: None) -> None:
        """Test upload_file with a PDF document."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import upload_file

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(doc_path=t.document.localpath)
        t.add_computed_column(file_id=upload_file(t.document.localpath))

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'
        pdf_file = pdf_files[0]

        validate_update_status(t.insert(document=pdf_file), 1)
        results = t.collect()

        file_id = results['file_id'][0]
        assert file_id is not None
        assert len(file_id) > 0

        print(f"\n{'=' * 70}")
        print("upload_file (PDF) Test Results")
        print(f"{'=' * 70}")
        print(f"Document Path: {results['doc_path'][0]}")
        print(f"File ID: {file_id}")
        print(f"{'=' * 70}\n")

    def test_upload_file_video(self, reset_db: None) -> None:
        """Test upload_file with a video."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import upload_file

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(video_path=t.video.localpath)
        t.add_computed_column(file_id=upload_file(t.video.localpath))

        video_files = get_video_files()
        assert len(video_files) > 0, 'No video files found'
        video_file = video_files[0]

        validate_update_status(t.insert(video=video_file), 1)
        results = t.collect()

        file_id = results['file_id'][0]
        assert file_id is not None
        assert len(file_id) > 0

        print(f"\n{'=' * 70}")
        print("upload_file (Video) Test Results")
        print(f"{'=' * 70}")
        print(f"Video Path: {results['video_path'][0]}")
        print(f"File ID: {file_id}")
        print(f"{'=' * 70}\n")

    def test_chat_completions_image(self, reset_db: None) -> None:
        """Test chat_completions with an uploaded image."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_file

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_file(t.image.localpath))
        t.add_computed_column(
            response=chat_completions(t.file_id, 'Describe this image briefly')
        )
        t.add_computed_column(description=t.response['choices'][0]['message']['content'])

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        # Check response structure
        response = results['response'][0]
        assert 'choices' in response
        assert len(response['choices']) > 0
        assert 'message' in response['choices'][0]
        assert 'content' in response['choices'][0]['message']

        # Check extracted text
        description = results['description'][0]
        assert description is not None
        assert len(description) > 0

        print(f"\n{'=' * 70}")
        print("chat_completions (Image) Test Results")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"File ID: {results['file_id'][0]}")
        print(f"Description: {description}")
        print(f"{'=' * 70}\n")

    def test_chat_completions_document(self, reset_db: None) -> None:
        """Test chat_completions with an uploaded PDF."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_file

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(doc_path=t.document.localpath)
        t.add_computed_column(file_id=upload_file(t.document.localpath))
        t.add_computed_column(
            response=chat_completions(t.file_id, 'What is the title of this document?')
        )
        t.add_computed_column(title=t.response['choices'][0]['message']['content'])

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'
        pdf_file = pdf_files[0]

        validate_update_status(t.insert(document=pdf_file), 1)
        results = t.collect()

        title = results['title'][0]
        assert title is not None
        assert len(title) > 0

        print(f"\n{'=' * 70}")
        print("chat_completions (Document) Test Results")
        print(f"{'=' * 70}")
        print(f"Document Path: {results['doc_path'][0]}")
        print(f"File ID: {results['file_id'][0]}")
        print(f"Title: {title}")
        print(f"{'=' * 70}\n")

    def test_chat_completions_video(self, reset_db: None) -> None:
        """Test chat_completions with an uploaded video."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_file

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(video_path=t.video.localpath)
        t.add_computed_column(file_id=upload_file(t.video.localpath))
        t.add_computed_column(
            response=chat_completions(t.file_id, 'Describe what happens in this video briefly')
        )
        t.add_computed_column(description=t.response['choices'][0]['message']['content'])

        video_files = get_video_files()
        assert len(video_files) > 0, 'No video files found'
        video_file = video_files[0]

        validate_update_status(t.insert(video=video_file), 1)
        results = t.collect()

        description = results['description'][0]
        assert description is not None
        assert len(description) > 0

        print(f"\n{'=' * 70}")
        print("chat_completions (Video) Test Results")
        print(f"{'=' * 70}")
        print(f"Video Path: {results['video_path'][0]}")
        print(f"File ID: {results['file_id'][0]}")
        print(f"Description: {description}")
        print(f"{'=' * 70}\n")

    def test_multiple_chat_completions(self, reset_db: None) -> None:
        """Test running multiple chat_completions on the same file_id."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_file

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_file(t.image.localpath))
        t.add_computed_column(
            subject_response=chat_completions(t.file_id, 'What is the main subject?')
        )
        t.add_computed_column(subject=t.subject_response['choices'][0]['message']['content'])
        t.add_computed_column(
            colors_response=chat_completions(t.file_id, 'What colors are prominent?')
        )
        t.add_computed_column(colors=t.colors_response['choices'][0]['message']['content'])
        t.add_computed_column(
            setting_response=chat_completions(t.file_id, 'Is this indoors or outdoors?')
        )
        t.add_computed_column(setting=t.setting_response['choices'][0]['message']['content'])

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        assert len(results['subject'][0]) > 0
        assert len(results['colors'][0]) > 0
        assert len(results['setting'][0]) > 0

        print(f"\n{'=' * 70}")
        print("Multiple chat_completions Test Results")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"File ID: {results['file_id'][0]}")
        print(f"Subject: {results['subject'][0]}")
        print(f"Colors: {results['colors'][0]}")
        print(f"Setting: {results['setting'][0]}")
        print(f"{'=' * 70}\n")

    def test_chat_completions_model_tier(self, reset_db: None) -> None:
        """Test chat_completions with different model tiers."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_file

        model = 'vlmrun-orion-1:fast'

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_file(t.image.localpath))
        t.add_computed_column(
            response=chat_completions(t.file_id, 'Describe this image briefly', model=model)
        )
        t.add_computed_column(description=t.response['choices'][0]['message']['content'])

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        # Check that model is reflected in response
        response = results['response'][0]
        assert response['model'] == model

        description = results['description'][0]
        assert len(description) > 0

        print(f"\n{'=' * 70}")
        print("chat_completions (Model Tier) Test Results")
        print(f"{'=' * 70}")
        print(f"Model: {model}")
        print(f"Response Model: {response['model']}")
        print(f"Description: {description}")
        print(f"{'=' * 70}\n")
