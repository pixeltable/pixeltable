import io
import os
from typing import List

import pytest
from PIL import Image

import pixeltable as pxt
from pixeltable.iterators import PdfPageExtractor


def find_pdfs(path: str, limit: int = 50, recursive: bool = True) -> List[str]:
    """Find PDF files in a directory up to a certain limit."""
    pdf_paths = []
    if recursive:
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
                    if len(pdf_paths) >= limit:
                        return pdf_paths
    else:
        for file in os.listdir(path):
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(path, file))
                if len(pdf_paths) >= limit:
                    break
    return pdf_paths


class TestPdfPageExtractor:
    @pytest.mark.parametrize("limit", [2])  # Only test a few PDFs to keep runtime low
    def test_pdf_page_extraction(self, limit: int) -> None:
        pdfs = find_pdfs("tests/data/documents", limit=limit)
        assert len(pdfs) > 0, "No PDF files found for testing."

        for doc_path in pdfs:
            extractor = PdfPageExtractor(document=doc_path)

            for chunk in extractor:
                assert 'page' in chunk
                assert 'text' in chunk
                assert 'image' in chunk

                # Test text is not empty
                assert isinstance(chunk["text"], str)
                assert len(chunk["text"]) > 0

                # Test image is a PIL Image
                assert isinstance(chunk["image"], Image.Image)

                # Convert image to bytes and reopen it
                img_io = io.BytesIO()
                chunk["image"].save(img_io, format="PNG")
                img_bytes = img_io.getvalue()
                assert len(img_bytes) > 0

                reopened_img = Image.open(io.BytesIO(img_bytes))
                assert reopened_img.size == chunk["image"].size
                break  # Only test the first page to keep the test fast

            extractor.close()
