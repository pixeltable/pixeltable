import io
import os
import itertools
from typing import List

import pytest
from PIL import Image

from pixeltable.iterators.document import DocumentSplitter

#
# pytest -v tests/test_pdf_page_images.py
#

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


class TestPdfPageImages:
    @pytest.mark.parametrize("limit", [2])  # keep runtime low
    def test_pdf_page_images(self, limit: int) -> None:
        pdfs = find_pdfs("tests/data/documents", limit=limit)
        assert pdfs, "No PDF files found for testing."

        for doc_path in pdfs:
            splitter = DocumentSplitter(
                document=doc_path,
                separators="page",
                metadata="page",
                include_page_image=True,
                page_image_dpi=150,  # lower for test speed
                page_image_format="png",
            )

            for chunk in itertools.islice(splitter, 3):
                assert "page" in chunk
                assert "text" in chunk
                assert isinstance(chunk["text"], str)
                assert len(chunk["text"]) > 0

                assert "image" in chunk
                if chunk["image"] is not None:
                    assert isinstance(chunk["image"], Image.Image)

                    # round-trip bytes check
                    img_io = io.BytesIO()
                    chunk["image"].save(img_io, format="PNG")
                    reopened = Image.open(io.BytesIO(img_io.getvalue()))
                    assert reopened.size == chunk["image"].size