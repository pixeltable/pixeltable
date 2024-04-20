import difflib
import itertools
import json
import re
from typing import List, Optional, Set

import ftfy
import pytest

import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.tests.utils import (
    get_audio_files,
    get_documents,
    get_image_files,
    get_video_files,
    skip_test_if_not_installed,
)
from pixeltable.type_system import DocumentType


def get_full_text_from_pdf(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    full_text_parts = []
    for page in doc:
        full_text_parts.append(page.get_text())

    full_text = '\n'.join(full_text_parts)
    return ftfy.fix_text(full_text)


def normalize(text: str):
    """ for comparison """
    res = re.sub(r'\s+', '', text)
    return res

def diff_snippet(text1: str, text2: str, diff_line_limit: Optional[int] = 20):
    diff = difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm='')
    if diff_line_limit is not None:
        snippet = [line for i, line in enumerate(diff) if i < diff_line_limit]
    else:
        snippet = list(diff)
    return '\n'.join(snippet)


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

    def test_pdf_splitter(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('fitz')
        skip_test_if_not_installed('tiktoken')

        file_paths = [p for p in self.valid_doc_paths() if p.endswith('.pdf')]

        assert len(file_paths) > 0
        for path in file_paths:
            full_text = normalize(get_full_text_from_pdf(path))

            level1 = ['', 'page', 'paragraph', 'sentence']
            level2 = ['', 'char_limit', 'token_limit']

            for sep1 in level1:
                for sep2 in level2:
                    chunk_limits = [10, 20, 100] if sep2 else [None]
                    for limit in chunk_limits:
                        iterator_args = {
                            'document': path,
                            'separators': ','.join([sep1, sep2]),
                            'metadata': 'page,bounding_box',
                        }
                        if sep2:
                            iterator_args['limit'] = limit
                            iterator_args['overlap'] = 0

                        chunks = list(DocumentSplitter(**iterator_args))
                        recovered_text = ' '.join([c['text'] for c in chunks])
                        diff_str = diff_snippet(full_text, normalize(recovered_text))
                        assert not diff_str, f'{iterator_args=}\n{diff_str}'
                        assert 'page' in chunks[0]
                        assert 'bounding_box' in chunks[0]

                        if sep1 not in ['']:
                            assert 'page' in chunks[0]
                            assert chunks[0]['page'] is not None

                        if sep1 not in ['', 'page']:
                            assert 'bounding_box' in chunks[0]
                            bounding_box = chunks[0]['bounding_box']
                            assert bounding_box is not None
                            assert 'x1' in bounding_box
                            assert bounding_box['x1'] is not None


    def test_doc_splitter(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('tiktoken')
        file_paths = [p for p in self.valid_doc_paths() if not p.endswith('.pdf')]
        cl = test_client
        doc_t = cl.create_table('docs', {'doc': DocumentType()})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        # run all combinations of (heading, paragraph, sentence) x (token_limit, char_limit, None)
        # and make sure they extract the same text in aggregate
        all_text_reference: Optional[str] = None  # all text as a single string; normalized
        headings_reference: Set[str] = {}  # headings metadata as a json-serialized string
        for sep1 in ['heading', 'paragraph', 'sentence']:
            for sep2 in ['', 'token_limit', 'char_limit']:
                chunk_limits = [10, 20, 100] if sep2 else [None]
                for limit in chunk_limits:
                    iterator_args = {
                        'document': doc_t.doc,
                        'separators': sep1 + (',' + sep2 if sep2 is not None else ''),
                        'metadata': 'title,heading,sourceline'
                    }
                    if sep2:
                        iterator_args['limit'] = limit
                        iterator_args['overlap'] = 0
                    chunks_t = cl.create_view(
                        f'chunks', doc_t, iterator_class=DocumentSplitter, iterator_args=iterator_args)
                    res = list(chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect())

                    if all_text_reference is None:
                        all_text_reference = normalize(''.join([r['text'] for r in res]))
                        headings_reference = set(json.dumps(r['heading']) for r in res)
                    else:
                        all_text = normalize(''.join([r['text'] for r in res]))
                        headings = set(json.dumps(r['heading']) for r in res)

                        diff = diff_snippet(all_text, all_text_reference)
                        assert not diff, f'{sep1}, {sep2}, {limit}\n{diff}'
                        assert headings == headings_reference, f'{sep1}, {sep2}, {limit}'
                        # TODO: verify chunk limit
                    cl.drop_table('chunks')

    def test_doc_splitter_headings(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('spacy')
        file_paths = [ p for p in self.valid_doc_paths() if not p.endswith('.pdf') ]
        cl = test_client
        doc_t = cl.create_table('docs', {'doc': DocumentType()})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        # verify that only the requested metadata is present in the view
        md_elements = ['title', 'heading', 'sourceline']
        md_tuples = list(itertools.chain.from_iterable(itertools.combinations(md_elements, i) for i in range(len(md_elements) + 1)))
        _ = [','.join(t) for t in md_tuples]
        for md_str in [','.join(t) for t in md_tuples]:
            iterator_args = {
                'document': doc_t.doc,
                'separators': 'sentence',
                'metadata': md_str
            }
            print(f'{md_str=}')
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
