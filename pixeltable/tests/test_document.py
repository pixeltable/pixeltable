from typing import Optional

import pixeltable as pxt
from pixeltable.type_system import DocumentType
from pixeltable.tests.utils import get_html_files


class TestAudio:
    def test_basic(self, test_client: pxt.Client) -> None:
        file_paths, _ = get_html_files()
        cl = test_client
        doc_t = cl.create_table('docs', {'html_doc': DocumentType()})
        status = doc_t.insert([{'html_doc': p} for p in file_paths])
        assert status.num_rows == len(file_paths)
        assert status.num_excs == 0
        stored_paths = doc_t.select(output=doc_t.html_doc.localpath).collect()['output']
        assert set(stored_paths) == set(file_paths)
