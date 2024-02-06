from typing import Optional, Set
import json
import re

import pixeltable as pxt
from pixeltable.type_system import DocumentType
from pixeltable.tests.utils import get_html_files
from pixeltable.iterators.document import HTMLDocumentSplitter


class TestDocument:
    def test_basic(self, test_client: pxt.Client) -> None:
        file_paths, _ = get_html_files()
        cl = test_client
        doc_t = cl.create_table('docs', {'html_doc': DocumentType()})
        status = doc_t.insert([{'html_doc': p} for p in file_paths])
        assert status.num_rows == len(file_paths)
        assert status.num_excs == 0
        stored_paths = doc_t.select(output=doc_t.html_doc.localpath).collect()['output']
        assert set(stored_paths) == set(file_paths)

    def test_html_splitter(self, test_client: pxt.Client) -> None:
        file_paths, _ = get_html_files()
        cl = test_client
        doc_t = cl.create_table('docs', {'html_doc': DocumentType()})
        status = doc_t.insert([{'html_doc': p} for p in file_paths])
        assert status.num_excs == 0

        def normalize(s: str) -> str:
            # remove whitespace
            res = re.sub(r'\s+', '', s)
            # remove non-ascii
            res = res.encode('ascii', 'ignore').decode()
            return res

        # run all combinations of (heading, paragraph, sentence) x (token_limit, char_limit, None)
        # and make sure they extract the same text in aggregate
        all_text_reference: Optional[str] = None
        headings_reference: Set[str] = {}
        for sep1 in ['heading', 'paragraph', 'sentence']:
            for sep2 in [None, 'token_limit', 'char_limit']:
                chunk_limits = [10, 20, 100] if sep2 is not None else [None]
                for limit in chunk_limits:
                    iterator_args = {
                        'document': doc_t.html_doc,
                        'separators': sep1 + (',' + sep2 if sep2 is not None else ''),
                        'metadata': 'title,headings,sourceline'
                    }
                    if sep2 is not None:
                        iterator_args['limit'] = limit
                        iterator_args['overlap'] = 0
                    chunks_t = cl.create_view(
                        f'chunks', doc_t, iterator_class=HTMLDocumentSplitter, iterator_args=iterator_args)
                    res = list(chunks_t.order_by(chunks_t.html_doc, chunks_t.pos).collect())

                    if all_text_reference is None:
                        all_text_reference = normalize(''.join([r['text'] for r in res]))
                        headings_reference = set(json.dumps(r['headings']) for r in res)
                    else:
                        all_text = normalize(''.join([r['text'] for r in res]))
                        headings = set(json.dumps(r['headings']) for r in res)

                        # for debugging
                        first_diff_index = next(
                            (i for i, (c1, c2) in enumerate(zip(all_text, all_text_reference)) if c1 != c2),
                            len(all_text) if len(all_text) != len(all_text_reference) else None)
                        if first_diff_index is not None:
                            a = all_text[max(0, first_diff_index - 10):first_diff_index + 10]
                            b = all_text_reference[max(0, first_diff_index - 10):first_diff_index + 10]

                        assert all_text == all_text_reference, f'{sep1}, {sep2}, {limit}'
                        assert headings == headings_reference, f'{sep1}, {sep2}, {limit}'
                        # TODO: verify chunk limit
                    cl.drop_table('chunks')
