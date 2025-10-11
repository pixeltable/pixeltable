import pytest

import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from tests import utils


class TestPdfExtraction:
    @pytest.mark.usefixtures('reset_db')  # ensures DB is reset between test runs
    def test_pdf_page_chunking(self):
        pdf_paths = [path for path in utils.get_documents() if path.endswith('.pdf')]
        assert pdf_paths, 'No PDF files found for testing.'

        # Create base table
        docs = pxt.create_table('pdf_docs', {'document': pxt.Document}, if_exists='replace_force')

        # Insert documents
        insert_result = docs.insert({'document': p} for p in pdf_paths)

        assert insert_result.num_rows == len(pdf_paths)
        assert insert_result.num_excs == 0

        pdf_page_iterator = DocumentSplitter.create(
            document=docs.document,
            separators='page',
            metadata='page',
            include_page_image=True,
            page_image_dpi=75,
            page_image_format='png',
        )

        # Create view
        chunks_view = pxt.create_view('pdf_page_chunks', docs, iterator=pdf_page_iterator)

        # Run query
        results_set = chunks_view.select(chunks_view.page, chunks_view.image, chunks_view.text).collect()
        results = list(results_set)

        # Validate outputs
        assert isinstance(results, list)
        assert all('page' in r and 'text' in r for r in results)
        assert all(r['text'] and isinstance(r['text'], str) for r in results)

        # Print chunk count and sample if needed
        print(f'Extracted {len(results)} page chunks from {len(pdf_paths)} PDFs')
        if len(results) > 0:
            print('Sample chunk:', results[0])

        # Describe (optional, for manual inspection)
        docs.describe()
        chunks_view.describe()
