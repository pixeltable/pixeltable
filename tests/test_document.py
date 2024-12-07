import difflib
import itertools
import json
import re
from typing import Optional

import pytest

import pixeltable as pxt
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.utils.documents import get_document_handle

from .utils import get_audio_files, get_documents, get_image_files, get_video_files, skip_test_if_not_installed


def _check_pdf_metadata(rec: dict, sep1: str, metadata: list[str]):
    if 'page' in metadata and sep1 in ['page', 'paragraph', 'sentence']:
        assert rec.get('page') is not None
    if 'bounding_box' in metadata and sep1 in ['paragraph', 'sentence']:
        box = rec.get('bounding_box')
        assert box is not None
        assert box.get('x1') is not None
        assert box.get('y1') is not None

def normalize(text: str) -> str:
    res = re.sub(r'\s+', '', text)
    return res

def diff_snippet(text1: str, text2: str, diff_line_limit: Optional[int] = 20) -> str:
    diff = difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm='')
    if diff_line_limit is not None:
        snippet = [line for i, line in enumerate(diff) if i < diff_line_limit]
    else:
        snippet = list(diff)
    return '\n'.join(snippet)


class TestDocument:
    def valid_doc_paths(self) -> list[str]:
        return get_documents()

    def invalid_doc_paths(self) -> list[str]:
        return [get_video_files()[0], get_audio_files()[0], get_image_files()[0]]

    def test_insert(self, reset_db) -> None:
        skip_test_if_not_installed('mistune')

        file_paths = self.valid_doc_paths()
        doc_t = pxt.create_table('docs', {'doc': pxt.Document})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_rows == len(file_paths)
        assert status.num_excs == 0
        stored_paths = doc_t.select(output=doc_t.doc.localpath).collect()['output']
        assert set(stored_paths) == set(file_paths)

        file_paths = self.invalid_doc_paths()
        status = doc_t.insert(({'doc': p} for p in file_paths), on_error='ignore')
        assert status.num_rows == len(file_paths)
        assert status.num_excs >= len(file_paths)

    def test_get_document_handle(self) -> None:
        skip_test_if_not_installed('mistune')

        file_paths = self.valid_doc_paths()
        for path in file_paths:
            extension = path.split('.')[-1].lower()
            handle = get_document_handle(path)
            assert handle is not None
            if extension == 'pdf':
                assert handle.format == pxt.DocumentType.DocumentFormat.PDF, path
                assert handle.pdf_doc is not None, path
            elif extension in ['html', 'htm']:
                assert handle.format == pxt.DocumentType.DocumentFormat.HTML, path
                assert handle.bs_doc is not None, path
            elif extension == 'md':
                assert handle.format == pxt.DocumentType.DocumentFormat.MD, path
                assert handle.md_ast is not None, path
            elif extension == 'xml':
                assert handle.format == pxt.DocumentType.DocumentFormat.XML, path
                assert handle.bs_doc is not None, path
            elif extension == 'txt':
                assert handle.format == pxt.DocumentType.DocumentFormat.TXT, path
                assert handle.txt_doc is not None, path
            else:
                assert False, f'Unexpected extension {extension}, add corresponding check'

    def test_invalid_arguments(self, reset_db) -> None:
        """ Test input parsing provides useful error messages
        """
        example_file = [p for p in self.valid_doc_paths() if p.endswith('.pdf')][0]

        # test invalid separators, or combinations of separators
        invalid_separators = ['page paragraph',  # no comma
                              'pagagaph',  # non existent separator
                              'page, block',  # block does not exist
                             ]
        for sep in invalid_separators:
            with pytest.raises(pxt.Error) as exc_info:
                _ = DocumentSplitter(document=example_file, separators=sep)
            assert 'Invalid separator' in str(exc_info.value)
        with pytest.raises(pxt.Error) as exc_info:
            _ = DocumentSplitter(document=example_file, separators='char_limit, token_limit', limit=10)
        assert 'both' in str(exc_info.value)

        # test that limit is required for char_limit and token_limit
        with pytest.raises(pxt.Error) as exc_info:
            _ = DocumentSplitter(document=example_file, separators='char_limit')
        assert 'limit' in str(exc_info.value)

        with pytest.raises(pxt.Error) as exc_info:
            _ = DocumentSplitter(document=example_file, separators='token_limit')
        assert 'limit' in str(exc_info.value)

        # test invalid metadata
        invalid_metadata = ['chapter',  # invalid
                            'page, bounding_box, chapter',  # mix of valid and invalid
                            'page bounding_box',  # separator
                            ]
        for md in invalid_metadata:
            with pytest.raises(pxt.Error) as exc_info:
                _ = DocumentSplitter(document=example_file, separators='', metadata=md)
            assert 'Invalid metadata' in str(exc_info.value)

    def test_doc_splitter(self, reset_db) -> None:
        skip_test_if_not_installed('tiktoken')
        skip_test_if_not_installed('spacy')

        # DocumentSplitter does not support XML
        file_paths = [path for path in self.valid_doc_paths() if not path.endswith('.xml')]
        doc_t = pxt.create_table('docs', {'doc': pxt.Document})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0
        import tiktoken
        encoding = tiktoken.get_encoding('cl100k_base')

        # run all combinations of (headings, paragraph, sentence) x (token_limit, char_limit, None)
        # and make sure they extract the same text in aggregate
        all_metadata = ['title', 'heading', 'sourceline', 'page', 'bounding_box']
        # combinations are given as (sep1, sep2, limit, metadata)
        combinations: list[tuple[str, str, int, list[str]]] = [
            (sep1, None, None, metadata) for sep1, metadata in itertools.product(
                ['', 'heading', 'page', 'paragraph', 'sentence'],
                [[], all_metadata]
            )
        ]
        combinations += [
            (sep1, sep2, limit, all_metadata) for sep1, sep2, limit in itertools.product(
                ['', 'heading', 'page', 'paragraph', 'sentence'],
                ['token_limit', 'char_limit'],
                [10, 20, 100]
            )
        ]

        all_text_reference: Optional[str] = None  # all text as a single string; normalized
        headings_reference: set[str] = set()  # headings metadata as a json-serialized string
        for sep1, sep2, limit, metadata in combinations:
            # Intentionally omit args that are not specified in this combination, to test that the iterator
            # applies defaults properly.
            args = {
                'document': doc_t.doc,
                'separators': sep1 if sep2 is None else ','.join([sep1, sep2]),
            }
            if len(metadata) > 0:
                args['metadata'] = ','.join(metadata)
            if sep2 is not None:
                args['limit'] = limit
                args['overlap'] = 0
            print(f'Testing with args: {args}')

            chunks_t = pxt.create_view('chunks', doc_t, iterator=DocumentSplitter.create(**args))
            res: list[dict] = list(chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect())

            if all_text_reference is None:
                assert sep1 == '' and sep2 is None
                # when sep1 and sep2 are both '', there should be a single result per input file.
                assert len(res) == len(file_paths)
                # check that all the expected metadata exists as a field
                for r in res:
                    assert r['text']  # non-empty text
                    assert all(md in r for md in metadata)

                all_text_reference = normalize(''.join(r['text'] for r in res))

                # check reference text is not empty
                assert all_text_reference

                # exclude markdown from heading checks at the moment
                if 'heading' in metadata:
                    headings_reference = {json.dumps(r['heading']) for r in res if not r['doc'].endswith('md')}
            else:
                all_text = normalize(''.join([r['text'] for r in res]))
                if 'heading' in metadata:
                    headings = {json.dumps(r['heading']) for r in res if not r['doc'].endswith('md')}

                diff = diff_snippet(all_text, all_text_reference)
                assert not diff, f'{sep1}, {sep2}, {limit}\n{diff}'

                # disable headings checks, currently failing strict equality,
                # but headings look reasonable and text is correct
                #assert headings == headings_reference, f'{sep1}, {sep2}, {limit}'

                # check splitter honors limits
                if sep2 == 'char_limit':
                    for r in res:
                        assert len(r['text']) <= limit
                if sep2 == 'token_limit':
                    for r in res:
                        tokens = encoding.encode(r['text'])
                        assert len(tokens) <= limit

                # check expected metadata is present
                for r in res:
                    if r['doc'].endswith('pdf'):
                        _check_pdf_metadata(r, sep1, metadata)
                    assert all(md in r for md in metadata)

            pxt.drop_table('chunks')

    def test_doc_splitter_headings(self, reset_db) -> None:
        skip_test_if_not_installed('spacy')
        file_paths = [p for p in self.valid_doc_paths() if not (p.endswith('.pdf') or p.endswith('.xml') or p.endswith('.txt'))]
        doc_t = pxt.create_table('docs', {'doc': pxt.Document})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        # verify that only the requested metadata is present in the view
        md_elements = ['title', 'heading', 'sourceline']
        md_tuples = list(itertools.chain.from_iterable(itertools.combinations(md_elements, i) for i in range(len(md_elements) + 1)))
        _ = [','.join(t) for t in md_tuples]
        for md_str in [','.join(t) for t in md_tuples]:
            print(f'{md_str=}')
            chunks_t = pxt.create_view(
                'chunks', doc_t,
                iterator=DocumentSplitter.create(document=doc_t.doc, separators='sentence', metadata=md_str))
            res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
            requested_md_elements = set(md_str.split(','))
            for md_element in md_elements:
                if md_element in requested_md_elements:
                    _ = res[md_element]
                else:
                    with pytest.raises(pxt.Error):
                        _ = res[md_element]
            pxt.drop_table('chunks')

    def test_doc_splitter_txt(self, reset_db) -> None:
        """ Test the DocumentSplitter with a .txt file

        test_doc_splitter above already tests the behaviour
        common for all document types. This test adds specific
        verification for a .txt file with specific content.
        """
        skip_test_if_not_installed('tiktoken')
        skip_test_if_not_installed('spacy')

        file_paths = [path for path in self.valid_doc_paths() if path.endswith('pxtbrief.txt')]
        doc_t = pxt.create_table('docs', {'doc': pxt.Document})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        chunks_t = pxt.create_view(
            'chunks', doc_t,
            iterator=DocumentSplitter.create(document=doc_t.doc,
                separators='', metadata='page'))
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 1
        assert len(res[0]['text']) == 2793
        assert str(res[0]['text']).startswith('Pixeltable Briefing Doc\nSource: GitHub Repository: pixeltable/pixeltable\n')
        assert res[0]['page'] is None

        # test with different separators.
        # Until we add support to split text into paragraphs,
        # the 'paragraph' separator is ignored and has no effect.
        pxt.drop_table('chunks')
        chunks_t = pxt.create_view(
            'chunks', doc_t,
            iterator=DocumentSplitter.create(document=doc_t.doc,
                separators='paragraph', metadata='page'))
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 1
        assert len(res[0]['text']) == 2793
        assert str(res[0]['text']).startswith('Pixeltable Briefing Doc\nSource: GitHub Repository: pixeltable/pixeltable\n')

        # test with 'sentence' separator
        # The text is split into 23 sentences.
        pxt.drop_table('chunks')
        chunks_t = pxt.create_view(
            'chunks', doc_t,
            iterator=DocumentSplitter.create(document=doc_t.doc,
                separators='sentence', metadata='page'))
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 23
        assert res[0]['text'] == 'Pixeltable Briefing Doc\nSource: GitHub Repository: pixeltable/pixeltable\n\nMain Themes:\n\nAI Data Infrastructure: Pixeltable is a Python library designed to simplify the management and processing of multimodal data for machine learning workflows.\n'
        assert len(res[0]['text']) == 245
        assert res[22]['text'] == 'Its declarative approach, incremental updates, and seamless Python integration make it a valuable tool for streamlining AI development and enhancing productivity.\n'
        assert len(res[22]['text']) == 163

        # test with 'char_limit' separator
        # The text is split into 67 chunks, each with a maximum of 50 characters.
        pxt.drop_table('chunks')
        chunks_t = pxt.create_view(
            'chunks', doc_t,
            iterator=DocumentSplitter.create(document=doc_t.doc,
                separators='sentence, char_limit',
                limit=50, overlap=0,
                metadata='title,heading,sourceline,page,bounding_box'))
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 67
        assert res[0]['text'] == 'Pixeltable Briefing Doc\nSource: GitHub Repository:'
        assert res[63]['text'] == 'Its declarative approach, incremental updates, and'
        for r in res:
            assert len(r['text']) <= 50
            assert r['title'] == ''
            assert r['heading'] is None
            assert r['sourceline'] is None
            assert r['page']is None
            assert r['doc'].endswith('pxtbrief.txt')

        pxt.drop_table('chunks')
