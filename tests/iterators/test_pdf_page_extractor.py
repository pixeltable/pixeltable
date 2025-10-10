import io
import os
import itertools
from typing import List

import pytest
from PIL import Image

from pixeltable.iterators.document import DocumentSplitter
from pixeltable.utils import documents
import tests.utils as utils

try:
    from IPython.display import display
except ImportError:
    display = print  # fallback: just print if not in notebook

#
# pytest -v tests/test_pdf_page_images.py
#

class TestPdfPageImages:
    @pytest.mark.parametrize("limit", [2])  # keep runtime low
    def test_pdf_page_images(self, limit: int) -> None:

        pdfs = [path for path in utils.get_documents() if path.endswith('.pdf')]
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

            for idx, chunk in enumerate(itertools.islice(splitter, 3), start=1):
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

                    # --- NEW: show image in widget ---
                    display(f"PDF: {os.path.basename(doc_path)}, page {chunk['page']}")
                    display(chunk["image"])
