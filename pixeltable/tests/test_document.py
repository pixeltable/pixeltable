import itertools
import json
import re
from typing import Optional, Set, List

import pytest

import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.tests.utils import get_documents, get_video_files, get_audio_files, get_image_files
from pixeltable.tests.utils import skip_test_if_not_installed
from pixeltable.type_system import DocumentType


class TestDocument:
    def valid_doc_paths(self) -> List[str]:
        return get_documents()

    def invalid_doc_paths(self) -> List[str]:
        return [get_video_files()[0], get_audio_files()[0], get_image_files()[0]]

    def test_insert(self, test_client: pxt.Client) -> None:
        file_paths = self.valid_doc_paths()
        cl = test_client
        doc_t = cl.create_table('docs', {'doc': DocumentType()})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_rows == len(file_paths)
        assert status.num_excs == 0
        stored_paths = doc_t.select(output=doc_t.doc.localpath).collect()['output']
        assert set(stored_paths) == set(file_paths)

        file_paths = self.invalid_doc_paths()
        status = doc_t.insert(({'doc': p} for p in file_paths), fail_on_exception=False)
        assert status.num_rows == len(file_paths)
        assert status.num_excs == len(file_paths)

    def test_doc_splitter(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('tiktoken')
        file_paths = self.valid_doc_paths()
        cl = test_client
        doc_t = cl.create_table('docs', {'doc': DocumentType()})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        def normalize(s: str) -> str:
            # remove whitespace
            res = re.sub(r'\s+', '', s)
            # remove non-ascii
            res = res.encode('ascii', 'ignore').decode()
            return res

        # run all combinations of (heading, paragraph, sentence) x (token_limit, char_limit, None)
        # and make sure they extract the same text in aggregate
        all_text_reference: Optional[str] = None  # all text as a single string; normalized
        headings_reference: Set[str] = {}  # headings metadata as a json-serialized string
        for sep1 in ['heading', 'paragraph', 'sentence']:
            for sep2 in [None, 'token_limit', 'char_limit']:
                chunk_limits = [10, 20, 100] if sep2 is not None else [None]
                for limit in chunk_limits:
                    iterator_args = {
                        'document': doc_t.doc,
                        'separators': sep1 + (',' + sep2 if sep2 is not None else ''),
                        'metadata': 'title,headings,sourceline'
                    }
                    if sep2 is not None:
                        iterator_args['limit'] = limit
                        iterator_args['overlap'] = 0
                    chunks_t = cl.create_view(
                        f'chunks', doc_t, iterator_class=DocumentSplitter, iterator_args=iterator_args)
                    res = list(chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect())

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

    def test_doc_splitter_headings(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('spacy')
        file_paths = self.valid_doc_paths()
        cl = test_client
        doc_t = cl.create_table('docs', {'doc': DocumentType()})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        # verify that only the requested metadata is present in the view
        md_elements = ['title', 'headings', 'sourceline']
        md_tuples = list(itertools.chain.from_iterable(itertools.combinations(md_elements, i) for i in range(len(md_elements) + 1)))
        _ = [','.join(t) for t in md_tuples]
        for md_str in [','.join(t) for t in md_tuples]:
            iterator_args = {
                'document': doc_t.doc,
                'separators': 'sentence',
                'metadata': md_str
            }
            chunks_t = cl.create_view(
                f'chunks', doc_t, iterator_class=DocumentSplitter, iterator_args=iterator_args)
            res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
            requested_md_elements = set(md_str.split(','))
            for md_element in md_elements:
                if md_element in requested_md_elements:
                    _ = res[md_element]
                else:
                    with pytest.raises(pxt.Error):
                        _ = res[md_element]
            cl.drop_table('chunks')
