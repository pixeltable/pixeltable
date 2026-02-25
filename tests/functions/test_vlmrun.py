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
    def test_upload_image(self, reset_db: None) -> None:
        """Test upload_image with an image."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import upload_image

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_image(t.image))

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        file_id = results['file_id'][0]
        assert file_id is not None
        assert len(file_id) > 0

        print(f"\n{'=' * 70}")
        print("upload_image Test Results")
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
        from pixeltable.functions.vlmrun import chat_completions, upload_image

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_image(t.image))

        # Build messages using the file_id
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe this image briefly'},
            {'type': 'input_file', 'file_id': t.file_id}
        ]}]
        t.add_computed_column(response=chat_completions(messages, model='vlmrun-orion-1:auto'))
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

        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What is the title of this document?'},
            {'type': 'input_file', 'file_id': t.file_id}
        ]}]
        t.add_computed_column(response=chat_completions(messages, model='vlmrun-orion-1:auto'))
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

    def test_chat_completions_model_tier(self, reset_db: None) -> None:
        """Test chat_completions with different model tiers."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_image

        model = 'vlmrun-orion-1:fast'

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(file_id=upload_image(t.image))

        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe this image briefly'},
            {'type': 'input_file', 'file_id': t.file_id}
        ]}]
        t.add_computed_column(response=chat_completions(messages, model=model))
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

    def test_image_artifact_retrieval(self, reset_db: None) -> None:
        """Test chat_completions with output_artifacts and get_image_artifact using hosted URL."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, get_image_artifact

        # Use VLM Run's hosted example image
        IMAGE_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/image.agent/lunch-skyscraper.jpg"

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})

        # Build messages using image_url directly (no upload needed)
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Blur all the faces in this image'},
            {'type': 'image_url', 'image_url': {'url': IMAGE_URL}}
        ]}]
        t.add_computed_column(response=chat_completions(messages, output_artifacts=['image']))

        # Extract session_id and image artifact ID
        t.add_computed_column(session_id=t.response['session_id'])
        t.add_computed_column(image_id=t.response['image'])

        # Retrieve the actual image artifact
        t.add_computed_column(blurred_image=get_image_artifact(
            t.image_id, session_id=t.session_id
        ))

        validate_update_status(t.insert(prompt='blur faces'), 1)
        results = t.collect()

        # Verify session_id was returned
        session_id = results['session_id'][0]
        assert session_id is not None, "session_id should be present in response"
        assert len(session_id) > 0, "session_id should not be empty"

        # Verify image artifact ID was returned (format: img_XXXXXX)
        image_id = results['image_id'][0]
        assert image_id is not None, "image artifact ID should be present"
        assert image_id.startswith('img_'), f"image ID should start with 'img_', got: {image_id}"

        # Verify we got an actual image back
        blurred_image = results['blurred_image'][0]
        assert blurred_image is not None, "blurred_image should not be None"

        print(f"\n{'=' * 70}")
        print("Image Artifact Retrieval Test Results")
        print(f"{'=' * 70}")
        print(f"Session ID: {session_id}")
        print(f"Image ID: {image_id}")
        print(f"Blurred Image: {type(blurred_image)} - {blurred_image.size if hasattr(blurred_image, 'size') else 'N/A'}")
        print(f"{'=' * 70}\n")

    def test_multi_image_artifact_retrieval(self, reset_db: None) -> None:
        """Test chat_completions with output_artifacts=['images'] for multiple image artifacts."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, get_image_artifact

        # Use VLM Run's hosted example image
        IMAGE_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/image.agent/lunch-skyscraper.jpg"

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})

        # Request multiple image outputs from a single image
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Create 3 different cropped versions of this image focusing on different people'},
            {'type': 'image_url', 'image_url': {'url': IMAGE_URL}}
        ]}]
        t.add_computed_column(response=chat_completions(messages, output_artifacts=['images']))

        t.add_computed_column(session_id=t.response['session_id'])
        t.add_computed_column(image_ids=t.response['images'])

        validate_update_status(t.insert(prompt='crop people'), 1)
        results = t.collect()

        session_id = results['session_id'][0]
        assert session_id is not None

        # Verify image_ids is a list
        image_ids = results['image_ids'][0]
        assert image_ids is not None, "image_ids should be present"
        assert isinstance(image_ids, list), f"image_ids should be a list, got: {type(image_ids)}"
        assert len(image_ids) > 0, "image_ids list should not be empty"

        # Verify each ID has the correct format
        for img_id in image_ids:
            assert img_id.startswith('img_'), f"image ID should start with 'img_', got: {img_id}"

        print(f"\n{'=' * 70}")
        print("Multi-Image Artifact Retrieval Test Results")
        print(f"{'=' * 70}")
        print(f"Session ID: {session_id}")
        print(f"Image IDs ({len(image_ids)}): {image_ids}")
        print(f"{'=' * 70}\n")

    def test_document_artifact_retrieval(self, reset_db: None) -> None:
        """Test chat_completions with document processing and artifact retrieval."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, get_document_artifact, upload_file

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(file_id=upload_file(t.document.localpath))

        # Request document redaction (returns a document artifact)
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Redact any sensitive information in this document'},
            {'type': 'input_file', 'file_id': t.file_id}
        ]}]
        t.add_computed_column(response=chat_completions(messages, output_artifacts=['document']))

        t.add_computed_column(session_id=t.response['session_id'])
        t.add_computed_column(doc_id=t.response['document'])
        t.add_computed_column(redacted_doc=get_document_artifact(t.doc_id, session_id=t.session_id))

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'

        validate_update_status(t.insert(document=pdf_files[0]), 1)
        results = t.collect()

        session_id = results['session_id'][0]
        assert session_id is not None

        doc_id = results['doc_id'][0]
        assert doc_id is not None
        assert doc_id.startswith('doc_'), f"document ID should start with 'doc_', got: {doc_id}"

        redacted_doc = results['redacted_doc'][0]
        assert redacted_doc is not None

        print(f"\n{'=' * 70}")
        print("Document Artifact Retrieval Test Results")
        print(f"{'=' * 70}")
        print(f"Session ID: {session_id}")
        print(f"Document ID: {doc_id}")
        print(f"Redacted Document: {redacted_doc}")
        print(f"{'=' * 70}\n")

    def test_chat_completions_toolsets(self, reset_db: None) -> None:
        """Test chat_completions with toolsets parameter."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_completions, upload_file

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(file_id=upload_file(t.document.localpath))

        # Use document toolset for processing PDFs
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Extract all text from this document'},
            {'type': 'input_file', 'file_id': t.file_id}
        ]}]
        t.add_computed_column(response=chat_completions(
            messages, model='vlmrun-orion-1:auto', toolsets=['document']
        ))
        t.add_computed_column(extracted=t.response['choices'][0]['message']['content'])

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'

        validate_update_status(t.insert(document=pdf_files[0]), 1)
        results = t.collect()

        # Verify response structure
        response = results['response'][0]
        assert 'choices' in response
        assert len(response['choices']) > 0

        # Verify extracted text is present
        extracted = results['extracted'][0]
        assert extracted is not None
        assert len(extracted) > 0

        print(f"\n{'=' * 70}")
        print("chat_completions (Toolsets) Test Results")
        print(f"{'=' * 70}")
        print(f"Toolsets: ['document']")
        print(f"Extracted text length: {len(extracted)}")
        print(f"Extracted (first 200 chars): {extracted[:200]}...")
        print(f"{'=' * 70}\n")
