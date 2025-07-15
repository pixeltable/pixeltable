import os
import pytest
import pixeltable as pxt
from typing import List
from pixeltable.iterators.pdf_page_extractor import PdfPageExtractor  # adjust path as needed
from pixeltable.catalog import Catalog

class TestPdfExtraction:
    
    def find_pdfs(self, path: str, limit: int = 50, recursive: bool = True) -> List[str]:
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

    @pytest.mark.usefixtures("reset_db")  # ensures DB is reset between test runs
    def test_pdf_page_chunking(self):
        pdf_paths = self.find_pdfs("tests/data/documents", limit=50)
        assert len(pdf_paths) > 0, "No PDF files found for testing."

        # Drop existing if any (safety)
        Catalog.get().drop_table('pdf_page_chunks', if_not_exists=True, force=True)
        Catalog.get().drop_table('pdf_docs', if_not_exists=True, force=True)

        # Create base table
        docs = pxt.create_table('pdf_docs', {'doc': pxt.String}, if_exists='replace_force')

        # Insert documents
        insert_result = docs.insert([{'doc': p} for p in pdf_paths])
        assert insert_result.num_rows == len(pdf_paths)
        assert insert_result.num_excs == 0

        # Create view
        chunks_view = pxt.create_view(
            'pdf_page_chunks',
            docs,
            iterator=PdfPageExtractor.create(document=docs.doc)
        )

        # Run query
        results_set = chunks_view.select(
            chunks_view.page,
            chunks_view.image,
            chunks_view.text
        ).collect()
        results = list(results_set)

        # Validate outputs
        assert isinstance(results, list)
        assert all('page' in r and 'text' in r for r in results)
        assert all(r['text'] and isinstance(r['text'], str) for r in results)

        # Print chunk count and sample if needed
        print(f"Extracted {len(results)} page chunks from {len(pdf_paths)} PDFs")
        if len(results) > 0:
            print("Sample chunk:", results[0])

        # Describe (optional, for manual inspection)
        docs.describe()
        chunks_view.describe()
