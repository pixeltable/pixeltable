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
    def test_chat_image(self, reset_db: None) -> None:
        """Test chat_image UDF with Pixeltable Image column."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_image

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(description=chat_image(t.image, 'Describe this image briefly'))

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()
        assert len(results['description'][0]) > 0

        print(f"\n{'=' * 70}")
        print("chat_image Test Results")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"Description: {results['description'][0]}")
        print(f"{'=' * 70}\n")

    def test_chat_document_with_pdf(self, reset_db: None) -> None:
        """Test chat_document UDF with Pixeltable Document column."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_document

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(doc_path=t.document.localpath)
        t.add_computed_column(summary=chat_document(t.document.localpath, 'Summarize this document briefly'))

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'
        pdf_file = pdf_files[0]

        validate_update_status(t.insert(document=pdf_file), 1)
        results = t.collect()
        assert len(results['summary'][0]) > 0

        print(f"\n{'=' * 70}")
        print("chat_document Test Results")
        print(f"{'=' * 70}")
        print(f"Document Path: {results['doc_path'][0]}")
        print(f"Summary: {results['summary'][0]}")
        print(f"{'=' * 70}\n")

    def test_upload_and_chat_with_file(self, reset_db: None) -> None:
        """Test two-stage video workflow: upload once, analyze multiple times."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_with_file, upload

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(video_path=t.video.localpath)

        # Stage 1: Upload video and store file_id
        t.add_computed_column(vlmrun_file_id=upload(t.video.localpath))

        # Stage 2: Multiple analyses using the same file_id
        t.add_computed_column(
            description=chat_with_file(t.vlmrun_file_id, 'Describe what happens in this video briefly')
        )
        t.add_computed_column(
            vehicles=chat_with_file(t.vlmrun_file_id, 'What vehicles are visible in this video?')
        )
        t.add_computed_column(
            location=chat_with_file(t.vlmrun_file_id, 'What city or country does this appear to be?')
        )

        video_files = get_video_files()
        assert len(video_files) > 0, 'No video files found'
        video_file = video_files[0]

        validate_update_status(t.insert(video=video_file), 1)
        results = t.collect()

        # Verify file_id was returned
        assert results['vlmrun_file_id'][0] is not None
        assert len(results['vlmrun_file_id'][0]) > 0

        # Verify all analyses were performed
        assert len(results['description'][0]) > 0
        assert len(results['vehicles'][0]) > 0
        assert len(results['location'][0]) > 0

        print(f"\n{'=' * 70}")
        print("upload + chat_with_file Test Results (Video)")
        print(f"{'=' * 70}")
        print(f"Video Path: {results['video_path'][0]}")
        print(f"VLM Run File ID: {results['vlmrun_file_id'][0]}")
        print(f"\nDescription: {results['description'][0]}")
        print(f"\nVehicles: {results['vehicles'][0]}")
        print(f"\nLocation: {results['location'][0]}")
        print(f"{'=' * 70}\n")

    def test_redact(self, reset_db: None) -> None:
        """Test redact UDF with image containing PII."""
        import os

        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import redact

        t = pxt.create_table('test_tbl', {'doc_path': pxt.String})
        t.add_computed_column(redacted_path=redact(t.doc_path))

        # Use the sample referral form with PII
        test_file = 'tests/data/documents/physical-therapy-referral.jpg'
        validate_update_status(t.insert(doc_path=test_file), 1)
        results = t.collect()

        # Verify redacted file path was returned
        redacted_path = results['redacted_path'][0]
        assert redacted_path is not None
        assert len(redacted_path) > 0

        # Verify file exists
        assert os.path.exists(redacted_path)

        print(f"\n{'=' * 70}")
        print("redact Test Results")
        print(f"{'=' * 70}")
        print(f"Original Path: {results['doc_path'][0]}")
        print(f"Redacted Path: {redacted_path}")
        print(f"Redacted File Size: {os.path.getsize(redacted_path):,} bytes")
        print(f"{'=' * 70}\n")

    def test_redact_custom_instructions(self, reset_db: None) -> None:
        """Test redact UDF with custom redaction instructions."""
        import os

        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import redact

        custom_instructions = 'Redact only names and phone numbers from this document.'

        t = pxt.create_table('test_tbl', {'doc_path': pxt.String})
        t.add_computed_column(
            redacted_path=redact(t.doc_path, instructions=custom_instructions)
        )

        test_file = 'tests/data/documents/physical-therapy-referral.jpg'
        validate_update_status(t.insert(doc_path=test_file), 1)
        results = t.collect()

        redacted_path = results['redacted_path'][0]
        assert redacted_path is not None
        assert os.path.exists(redacted_path)

        print(f"\n{'=' * 70}")
        print("redact (Custom Instructions) Test Results")
        print(f"{'=' * 70}")
        print(f"Original Path: {results['doc_path'][0]}")
        print(f"Instructions: {custom_instructions}")
        print(f"Redacted Path: {redacted_path}")
        print(f"Redacted File Size: {os.path.getsize(redacted_path):,} bytes")
        print(f"{'=' * 70}\n")

    def test_upload_with_image(self, reset_db: None) -> None:
        """Test upload() with image files."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import upload

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        t.add_computed_column(vlmrun_file_id=upload(t.image.localpath))

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        # Verify file_id was returned (UUID format)
        vlmrun_file_id = results['vlmrun_file_id'][0]
        assert vlmrun_file_id is not None
        assert len(vlmrun_file_id) > 0

        print(f"\n{'=' * 70}")
        print("upload (Image) Test Results")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"VLM Run File ID: {vlmrun_file_id}")
        print(f"{'=' * 70}\n")

    def test_upload_with_pdf(self, reset_db: None) -> None:
        """Test upload() with PDF documents."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import upload

        t = pxt.create_table('test_tbl', {'document': pxt.Document})
        t.add_computed_column(doc_path=t.document.localpath)
        t.add_computed_column(vlmrun_file_id=upload(t.document.localpath))

        pdf_files = [f for f in get_documents() if f.endswith('.pdf')]
        assert len(pdf_files) > 0, 'No PDF files found in test documents'
        pdf_file = pdf_files[0]

        validate_update_status(t.insert(document=pdf_file), 1)
        results = t.collect()

        vlmrun_file_id = results['vlmrun_file_id'][0]
        assert vlmrun_file_id is not None
        assert len(vlmrun_file_id) > 0

        print(f"\n{'=' * 70}")
        print("upload (PDF) Test Results")
        print(f"{'=' * 70}")
        print(f"Document Path: {results['doc_path'][0]}")
        print(f"VLM Run File ID: {vlmrun_file_id}")
        print(f"{'=' * 70}\n")

    def test_multiple_analyses_same_file_id(self, reset_db: None) -> None:
        """Test that multiple analyses can run on the same file_id without re-upload."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_with_file, upload

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)

        # Upload once
        t.add_computed_column(vlmrun_file_id=upload(t.image.localpath))

        # Run multiple different analyses on the same file_id
        t.add_computed_column(
            subject=chat_with_file(t.vlmrun_file_id, 'What is the main subject of this image?')
        )
        t.add_computed_column(
            colors=chat_with_file(t.vlmrun_file_id, 'What colors are prominent in this image?')
        )
        t.add_computed_column(
            setting=chat_with_file(t.vlmrun_file_id, 'Is this image indoors or outdoors?')
        )
        t.add_computed_column(
            objects=chat_with_file(t.vlmrun_file_id, 'List all objects visible in this image.')
        )
        t.add_computed_column(
            mood=chat_with_file(t.vlmrun_file_id, 'What is the mood or atmosphere of this image?')
        )

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()

        # Verify file_id was returned
        vlmrun_file_id = results['vlmrun_file_id'][0]
        assert vlmrun_file_id is not None
        assert len(vlmrun_file_id) > 0

        # All 5 analyses should return non-empty results
        assert len(results['subject'][0]) > 0
        assert len(results['colors'][0]) > 0
        assert len(results['setting'][0]) > 0
        assert len(results['objects'][0]) > 0
        assert len(results['mood'][0]) > 0

        # Print all results for verification
        print(f"\n{'=' * 70}")
        print("Multi-Analysis Test Results (Single Upload, 5 Analyses)")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"VLM Run File ID: {vlmrun_file_id}")
        print(f"\n1. Subject: {results['subject'][0]}")
        print(f"\n2. Colors: {results['colors'][0]}")
        print(f"\n3. Setting: {results['setting'][0]}")
        print(f"\n4. Objects: {results['objects'][0]}")
        print(f"\n5. Mood: {results['mood'][0]}")
        print(f"{'=' * 70}\n")

    def test_chat_image_model_tier(self, reset_db: None) -> None:
        """Test chat_image with different model tiers."""
        skip_test_if_not_installed('vlmrun')
        skip_test_if_no_client('vlmrun')
        from pixeltable.functions.vlmrun import chat_image

        model = 'vlmrun-orion-1:fast'

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        t.add_computed_column(image_path=t.image.localpath)
        # Test with 'fast' model tier
        t.add_computed_column(
            description=chat_image(t.image, 'Describe this image briefly', model=model)
        )

        image_file = get_image_files()[0]
        validate_update_status(t.insert(image=image_file), 1)
        results = t.collect()
        assert len(results['description'][0]) > 0

        print(f"\n{'=' * 70}")
        print("chat_image (Model Tier) Test Results")
        print(f"{'=' * 70}")
        print(f"Image Path: {results['image_path'][0]}")
        print(f"Model: {model}")
        print(f"Description: {results['description'][0]}")
        print(f"{'=' * 70}\n")
