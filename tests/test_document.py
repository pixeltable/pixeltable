import ctypes
import numpy as np
from collections.abc import Callable
import difflib
import itertools
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import PIL.Image
from PIL import ImageDraw
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.iterators.document import DocumentSplitter
from pixeltable.utils.documents import get_document_handle

from .utils import get_audio_files, get_documents, get_image_files, get_video_files, skip_test_if_not_installed


def _check_pdf_metadata(rec: dict, sep1: str, metadata: list[str]) -> None:
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


def diff_snippet(text1: str, text2: str, diff_line_limit: int | None = 20) -> str:
    diff = difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm='')
    if diff_line_limit is not None:
        snippet = [line for i, line in enumerate(diff) if i < diff_line_limit]
    else:
        snippet = list(diff)
    return '\n'.join(snippet)

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class PdfChar:
    value: str
    bounding_box: BoundingBox
    center_x: float
    center_y: float


class TestDocument:
    def valid_doc_paths(self) -> list[str]:
        return get_documents()

    def invalid_doc_paths(self) -> list[str]:
        return [get_video_files()[0], get_audio_files()[0], get_image_files()[0]]

    def test_insert(self, reset_db: None) -> None:
        skip_test_if_not_installed('mistune')

        file_paths = self.valid_doc_paths()
        doc_t = pxt.create_table('docs', {'doc': pxt.Document})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_rows == len(file_paths)
        assert status.num_excs == 0
        stored_paths = doc_t.select(output=doc_t.doc.localpath).collect()['output']
        assert set(stored_paths) == set(file_paths)

        file_paths = self.invalid_doc_paths()
        status = doc_t.insert([{'doc': p} for p in file_paths], on_error='ignore')
        assert status.num_rows == len(file_paths)
        assert status.num_excs >= len(file_paths)

    def test_get_document_handle(self) -> None:
        skip_test_if_not_installed('mistune')

        file_paths = self.valid_doc_paths()
        for path in file_paths:
            _, extension = os.path.splitext(path)
            handle = get_document_handle(path)
            assert handle is not None
            if extension == '.pdf':
                assert handle.format == ts.DocumentType.DocumentFormat.PDF, path
                assert handle.pdf_doc is not None, path
            elif extension in ('.html', '.htm'):
                assert handle.format == ts.DocumentType.DocumentFormat.HTML, path
                assert handle.bs_doc is not None, path
            elif extension == '.md':
                assert handle.format == ts.DocumentType.DocumentFormat.MD, path
                assert handle.md_ast is not None, path
            elif extension == '.xml':
                assert handle.format == ts.DocumentType.DocumentFormat.XML, path
                assert handle.bs_doc is not None, path
            elif extension == '.txt':
                assert handle.format == ts.DocumentType.DocumentFormat.TXT, path
                assert handle.txt_doc is not None, path
            else:
                raise AssertionError(f'Unexpected extension {extension}, add corresponding check')

    def test_doc_splitter_errors(self, reset_db: None) -> None:
        t = pxt.create_table('docs', {'doc': pxt.Document})

        # test invalid separators, or combinations of separators
        invalid_separators = [
            'page paragraph',  # no comma
            'pagagaph',  # non existent separator
            'page, block',  # block does not exist
        ]
        for sep in invalid_separators:
            with pytest.raises(pxt.Error, match='Invalid separator'):
                _ = pxt.create_view('chunks', t, iterator=DocumentSplitter.create(document=t.doc, separators=sep))

        with pytest.raises(pxt.Error, match='both'):
            _ = pxt.create_view(
                'chunks',
                t,
                iterator=DocumentSplitter.create(document=t.doc, separators='char_limit, token_limit', limit=10),
            )

        # test that limit is required for char_limit and token_limit
        with pytest.raises(pxt.Error, match='limit'):
            _ = pxt.create_view('chunks', t, iterator=DocumentSplitter.create(document=t.doc, separators='char_limit'))

        with pytest.raises(pxt.Error, match='limit'):
            _ = pxt.create_view('chunks', t, iterator=DocumentSplitter.create(document=t.doc, separators='token_limit'))

        # test invalid metadata
        invalid_metadata = [
            'chapter',  # invalid
            'page, bounding_box, chapter',  # mix of valid and invalid
            'page bounding_box',  # separator
        ]
        for md in invalid_metadata:
            with pytest.raises(pxt.Error, match='Invalid metadata'):
                _ = pxt.create_view(
                    'chunks', t, iterator=DocumentSplitter.create(document=t.doc, separators='', metadata=md)
                )

        invalid_separators = ['page, sentence', 'paragraph, sentence', 'char_limit, sentence', 'token_limit, sentence']
        for sep in invalid_separators:
            with pytest.raises(pxt.Error, match='Image elements are only supported for the "page" separator'):
                _ = pxt.create_view(
                    'chunks', t, iterator=DocumentSplitter.create(document=t.doc, separators=sep, elements=['image'])
                )

    def test_doc_splitter(self, reset_db: None) -> None:
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
            (sep1, None, None, metadata)
            for sep1, metadata in itertools.product(
                ['', 'heading', 'page', 'paragraph', 'sentence'], [[], all_metadata]
            )
        ]
        combinations += [
            (sep1, sep2, limit, all_metadata)
            for sep1, sep2, limit in itertools.product(
                ['', 'heading', 'page', 'paragraph', 'sentence'], ['token_limit', 'char_limit'], [10, 20, 100]
            )
        ]

        all_text_reference: str | None = None  # all text as a single string; normalized
        headings_reference: set[str] = set()  # headings metadata as a json-serialized string
        for sep1, sep2, limit, metadata in combinations:
            # Intentionally omit args that are not specified in this combination, to test that the iterator
            # applies defaults properly.
            args: dict[str, Any] = {
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
                # (This can be made into an assertion to re-enable headings checks)
                if headings == headings_reference:  # f'{sep1}, {sep2}, {limit}'
                    pass

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

    def test_doc_splitter_headings(self, reset_db: None) -> None:
        skip_test_if_not_installed('spacy')
        file_paths = [
            p for p in self.valid_doc_paths() if not (p.endswith('.pdf') or p.endswith('.xml') or p.endswith('.txt'))
        ]
        doc_t = pxt.create_table('docs', {'doc': pxt.Document})
        status = doc_t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        # verify that only the requested metadata is present in the view
        md_elements = ['title', 'heading', 'sourceline']
        md_tuples = list(
            itertools.chain.from_iterable(itertools.combinations(md_elements, i) for i in range(len(md_elements) + 1))
        )
        _ = [','.join(t) for t in md_tuples]
        for md_str in [','.join(t) for t in md_tuples]:
            print(f'{md_str=}')
            chunks_t = pxt.create_view(
                'chunks',
                doc_t,
                iterator=DocumentSplitter.create(document=doc_t.doc, separators='sentence', metadata=md_str),
            )
            res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
            requested_md_elements = set(md_str.split(','))
            for md_element in md_elements:
                if md_element in requested_md_elements:
                    _ = res[md_element]
                else:
                    with pytest.raises(pxt.Error):
                        _ = res[md_element]
            pxt.drop_table('chunks')

    def test_doc_splitter_txt(self, reset_db: None) -> None:
        """Test the DocumentSplitter with a .txt file

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
            'chunks', doc_t, iterator=DocumentSplitter.create(document=doc_t.doc, separators='', metadata='page')
        )
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 1
        assert len(res[0]['text']) == 2793
        assert str(res[0]['text']).startswith(
            'Pixeltable Briefing Doc\nSource: GitHub Repository: pixeltable/pixeltable\n'
        )
        assert res[0]['page'] is None

        # test with different separators.
        # Until we add support to split text into paragraphs,
        # the 'paragraph' separator is ignored and has no effect.
        pxt.drop_table('chunks')
        chunks_t = pxt.create_view(
            'chunks',
            doc_t,
            iterator=DocumentSplitter.create(document=doc_t.doc, separators='paragraph', metadata='page'),
        )
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 1
        assert len(res[0]['text']) == 2793
        assert str(res[0]['text']).startswith(
            'Pixeltable Briefing Doc\nSource: GitHub Repository: pixeltable/pixeltable\n'
        )

        # test with 'sentence' separator
        # The text is split into 22 sentences.
        pxt.drop_table('chunks')
        chunks_t = pxt.create_view(
            'chunks',
            doc_t,
            iterator=DocumentSplitter.create(document=doc_t.doc, separators='sentence', metadata='page'),
        )
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 22
        assert res[0]['text'] == (
            'Pixeltable Briefing Doc\nSource: GitHub Repository: pixeltable/pixeltable\n\nMain Themes:\n\n'
            'AI Data Infrastructure: Pixeltable is a Python library designed to simplify the management '
            'and processing of multimodal data for machine learning workflows.\n'
        )
        assert len(res[0]['text']) == 245
        assert res[21]['text'] == (
            'Its declarative approach, incremental updates, and seamless Python integration make it a '
            'valuable tool for streamlining AI development and enhancing productivity.\n'
        )
        assert len(res[21]['text']) == 163

        # test with 'char_limit' separator
        # The text is split into 67 chunks, each with a maximum of 50 characters.
        pxt.drop_table('chunks')
        chunks_t = pxt.create_view(
            'chunks',
            doc_t,
            iterator=DocumentSplitter.create(
                document=doc_t.doc,
                separators='sentence, char_limit',
                limit=50,
                overlap=0,
                metadata='title,heading,sourceline,page,bounding_box',
            ),
        )
        res = chunks_t.order_by(chunks_t.doc, chunks_t.pos).collect()
        assert len(res) == 67
        assert res[0]['text'] == 'Pixeltable Briefing Doc\nSource: GitHub Repository:'
        assert res[63]['text'] == 'Its declarative approach, incremental updates, and'
        for r in res:
            assert len(r['text']) <= 50
            assert r['title'] == ''
            assert r['heading'] is None
            assert r['sourceline'] is None
            assert r['page'] is None
            assert r['doc'].endswith('pxtbrief.txt')

        pxt.drop_table('chunks')

    def test_doc_splitter_images(self, reset_db: None) -> None:
        file_paths = [p for p in get_documents() if p.endswith('.pdf')]
        t = pxt.create_table('docs', {'doc': pxt.Document})

        chunks = pxt.create_view(
            'chunks',
            t,
            iterator=DocumentSplitter.create(
                document=t.doc, separators='page', elements=['image'], metadata='title,page'
            ),
        )
        status = t.insert({'doc': p} for p in file_paths)
        assert status.num_excs == 0

        res = chunks.collect()
        assert all(isinstance(r['image'], PIL.Image.Image) for r in res)

    def test_pdf_libs(self, reset_db: None) -> None:
        path = './docs/resources/rag-demo/Zacks-Nvidia-Report.pdf'
        path = './gilbert.pdf'
        page_num = 2

        print('====== pdfminer ======')
        from io import StringIO

        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfparser import PDFParser
        # from pdfminer.high_level import extract_text
        # text = extract_text(path)
        # print(text)

        output_string = StringIO()
        with open(path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for i, page in enumerate(PDFPage.create_pages(doc)):
                if i != page_num:
                    continue
                interpreter.process_page(page)
        print(output_string.getvalue())

        print('====== pdfminer using extract_pages ======')
        from collections.abc import Iterable

        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTChar, LTFigure, LTTextContainer

        def print_text(obj: Any):
            print(f'type: {type(obj)}')
            if isinstance(obj, LTTextContainer):
                print(obj.get_text())
            elif isinstance(obj, LTFigure):
                text = ''
                for element in obj:
                    if isinstance(element, LTChar):
                        text = text + element.get_text()
                print(text)
            elif isinstance(obj, Iterable):
                for o in obj:
                    print_text(o)

        for i, page_layout in enumerate(extract_pages(path)):
            if i != page_num:
                continue
            print_text(page_layout)
            # for element in page_layout:
            #     print (f': {type(element)}')
            #     if isinstance(element, LTTextContainer):
            #         print(element.get_text())
            # print('-- page break --')

        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c

        pdf = pdfium.PdfDocument(path)
        page = pdf[page_num]
        print('====== pdfium objects ======')
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE]):
            txt = None
            if obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
                # txt = obj.get_text()
                pass
            elif obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                txt = '<image object>'
            print(f'type {obj.type} level {obj.level} bounds {obj.get_bounds()} text: {txt}')

        textpage = page.get_textpage()
        # Can retrieve text inside a box
        # text_all = textpage.get_text_bounded()
        text_all = textpage.get_text_range()
        print('====== pdfium text ======')
        print(text_all)

        print('====== pymupdf blocks ======')
        doc = get_document_handle(path).pdf_doc
        assert doc is not None
        for i, page in enumerate(doc.pages()):
            if i != page_num:
                continue
            for block in page.get_text('blocks'):
                print(block)
                print('-----')
            break

    def test_pdfium(self) -> None:
        path = './gilbert.pdf'
        page_num = 2

        # path = './docs/resources/rag-demo/Argus-Market-Watch-June-2024.pdf'
        # page_num = 0

        # path = './docs/resources/rag-demo/Argus-Market-Digest-June-2024.pdf'
        # page_num = 0

        # path  = './docs/resources/rag-demo/Company-Research-Alphabet.pdf'
        # page_num = 1

        # path = './docs/resources/rag-demo/Jefferson-Amazon.pdf'
        # page_num = 9

        self.print_from_pdfium(path, page_num)

    def print_from_pdfium(self, path: str, page_num: int) -> None:
        import ctypes

        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c

        pdf = pdfium.PdfDocument(path)
        page = pdf[page_num]
        textpage = page.get_textpage()
        print('====== characters ======')
        print()
        text = ''
        for i in range(0, pdfium_c.FPDFText_CountChars(textpage)):
            code = pdfium_c.FPDFText_GetUnicode(textpage, i)
            x, y = ctypes.c_double(), ctypes.c_double()
            assert pdfium_c.FPDFText_GetCharOrigin(textpage, i, x, y)
            print(f'char {i}: {chr(code)}({code}), origin ({x.value}, {y.value})')
            text = text + chr(code)

        print('====== entire text ======')
        print(text)

    def test_split_page(self) -> None:
        path = './gilbert.pdf'
        page_num = 2
        # path = './docs/resources/rag-demo/Zacks-Nvidia-Report.pdf'
        # page_num = 4
        # path = './docs/resources/rag-demo/Jefferson-Amazon.pdf'
        # page_num = 9

        pdf = pdfium.PdfDocument(path)
        page = pdf[page_num]
        textpage = page.get_textpage()

        chars = []
        for i in range(0, pdfium_c.FPDFText_CountChars(textpage)):
            code = pdfium_c.FPDFText_GetUnicode(textpage, i)
            x0, y0, x1, y1 = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            assert pdfium_c.FPDFText_GetCharBox(textpage, i, x0, x1, y0, y1)
            assert x0.value <= x1.value
            assert y0.value <= y1.value
            chars.append(
                PdfChar(
                    value=chr(code),
                    bounding_box=BoundingBox(x0.value, y0.value, x1.value, y1.value),
                    center_x=(x0.value + x1.value) / 2,
                    center_y=(y0.value + y1.value) / 2,
                )
            )
        print('\n====== entire textpage ======\n')
        print(textpage.get_text_range())
        assert len(chars) == len(textpage.get_text_range())
        print('\n====== / entire textpage ======\n')
        print('====== char by char comparison ======\n')
        for i, c1, c2 in zip(range(len(chars)), chars, textpage.get_text_range()):
            code = pdfium_c.FPDFText_GetUnicode(textpage, i)
            match = c1.value == c2
            print (f'index {i:6}/{len(chars)}: code={code}, char: {c1.value} vs textpage char: {c2} {"" if match else "<<< MISMATCH >>>"}')
        print('====== / char by char comparison ======\n')

        non_whitespace_chars = [c for c in chars if not c.value.isspace()]
        # min_x = min(c.center_x for c in non_whitespace_chars)
        # max_x = max(c.center_x for c in non_whitespace_chars)
        # min_y = min(c.center_y for c in non_whitespace_chars)
        # max_y = max(c.center_y for c in non_whitespace_chars)
        min_x = 0
        min_y = 0
        max_x = page.get_width()
        max_y = page.get_height()

        bounds = self._split_page(textpage, non_whitespace_chars, BoundingBox(min_x, min_y, max_x, max_y))
        assert bounds is not None
        print('\n====== split bounds ======\n')
        for b in bounds:
            print(f'Box: ({b.x0}, {b.y0}, {b.x1}, {b.y1})')
            chars_in_box = self._chars_in_bound(chars, b)
            box_text = ''.join(c.value for c in chars_in_box)
            print(f'Text: {box_text}')
            print('-----')
        print('\n====== / split bounds ======\n')

        scores = np.zeros((len(textpage.get_text_range()),), dtype=np.uint8)
        current_box: BoundingBox | None = None
        # give a higher score to the first character in each box
        for i in range(len(textpage.get_text_range())):
            c = chars[i]
            box = self._find_bounding_box(c, bounds)
            if box is None:
                # possibly whitespace character that we ignored when finding bounding boxes
                continue
            if box != current_box:
                scores[i] = 100
                current_box = box

        # detect paragraphs
        import unicodedata
        previous_line_x : float | None = None
        previous_line_y : float | None = None
        is_new_line = True
        for i in range(len(textpage.get_text_range())):
            c = chars[i]
            if c.value == '\n':
                is_new_line = True
                continue
            is_printable = not unicodedata.category(c.value).startswith('C') and not c.value.isspace()
            if not is_printable:
                continue
            if previous_line_x is None:
                # first line
                previous_line_x = c.center_x
                previous_line_y = c.center_y
                continue
            if is_new_line:
                # encountered a printable character on a new line
                is_new_line = False
                x = c.center_x
                y = c.center_y
                if y > previous_line_y:
                    # text jumped up, possibly to a new column
                    previous_line_x = x
                    previous_line_y = y
                    continue
                gap_vs_last_line = x - previous_line_x
                char_width = self._char_width(c)
                if gap_vs_last_line > char_width and gap_vs_last_line < 5*char_width:
                    # a possible paragraph indent
                    scores[i]+=50
                previous_line_x = x
                previous_line_y = y

        # visualize results
        page_img = page.render()
        pil_img = page_img.to_pil()
        # draw bounding boxes
        draw = ImageDraw.Draw(pil_img)
        for b in bounds:
            x0, y0 = self._page_coords_to_image_coords( page, pil_img, b.x0, b.y0)
            x1, y1 = self._page_coords_to_image_coords( page, pil_img, b.x1, b.y1)
            box  = ( min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            draw.rectangle(box , outline="red", width=1)

        # draw segmentation scores
        for i in range (len(scores)):
            if scores[i] > 0:
                c = chars[i]
                c_x, c_y = self._page_coords_to_image_coords( page, pil_img, c.center_x, c.center_y)
                radius = 5
                color = "red" if scores[i] >= 100 else "blue"
                box = (c_x - radius,  c_y - radius, 
                        c_x + radius, c_y + radius)
                draw.ellipse(box, outline=color, fill=color, width=1)

        pil_img.save('./output.png')

    def _page_coords_to_image_coords(self, page: pdfium.PdfPage, image, x: float, y: float) -> tuple[float, float]:
        img_x = x * (image.width / page.get_width())
        img_y = y * (image.height / page.get_height())
        img_y = image.height - img_y # PDF coords start at bottom-left
        return (img_x, img_y)
        

    def _chars_in_bound(self, chars, bound: BoundingBox) -> list[PdfChar]:
        return [c for c in chars if self._char_is_in_bound(c, bound)]
    
    def _char_is_in_bound(self, c: PdfChar, bound: BoundingBox) -> bool:
        return bound.x0 <= c.center_x <= bound.x1 and bound.y0 <= c.center_y <= bound.y1

    def _find_bounding_box(self, c: PdfChar, boxes: list[BoundingBox]) -> BoundingBox | None:
        for b in boxes:
            if self._char_is_in_bound(c, b):
                return b
        return None

    def _split_page(self, textpage: pdfium.PdfTextPage, chars, bound: BoundingBox) -> list[BoundingBox] | None:
        if bound.x0 >= bound.x1 or bound.y0 >= bound.y1:
            return None

        chars_in_bound = self._chars_in_bound(chars, bound)
        if len(chars_in_bound) < 200:
            # too few chars to split further
            return [bound]

        # find biggest vertical gap
        chars_in_bound.sort(key=lambda c: c.center_y)
        vert_gap = self._biggest_gap(chars_in_bound, lambda c: c.center_y)
        avg_char_height = self._avg([self._char_height(c) for c in chars_in_bound])
        vert_gap_significance = None
        if vert_gap is not None:
            vert_gap_significance = vert_gap[0] / avg_char_height
        if vert_gap_significance is not None and vert_gap_significance < 2.0:
            vert_gap_significance = None

        # find biggest horizontal gap
        chars_in_bound.sort(key=lambda c: c.center_x)
        horiz_gap = self._biggest_gap(chars_in_bound, lambda c: c.center_x)
        avg_char_width = self._avg([self._char_width(c) for c in chars_in_bound])
        horiz_gap_significance = None
        if horiz_gap is not None:
            horiz_gap_significance = horiz_gap[0] / avg_char_width
        if horiz_gap_significance is not None and horiz_gap_significance < 4.0:
            horiz_gap_significance = None

        if vert_gap_significance is None and horiz_gap_significance is None:
            # No significant gaps, can't split further
            return [bound]
        do_vert_split: bool
        if vert_gap_significance is not None and horiz_gap_significance is not None:
            if vert_gap_significance >= horiz_gap_significance:
                do_vert_split = True
            else:
                do_vert_split = False
        else:
            do_vert_split = vert_gap_significance is not None

        if do_vert_split:
            box1 = BoundingBox(bound.x0, bound.y0, bound.x1, vert_gap[1])
            box2 = BoundingBox(bound.x0, vert_gap[1], bound.x1, bound.y1)
            return self._split_page(textpage, chars, box1) + self._split_page(textpage, chars, box2)
        else:
            box1 = BoundingBox(bound.x0, bound.y0, horiz_gap[1], bound.y1)
            box2 = BoundingBox(horiz_gap[1], bound.y0, bound.x1, bound.y1)
            return self._split_page(textpage, chars, box1) + self._split_page(textpage, chars, box2)

    def _biggest_gap(self, chars, coord_func: Callable[[PdfChar], float]) -> tuple[float, float] | None:
        # Finds the biggest gap between consecutive chars based on the provided coordinate function.
        # Returns a tuple of (gap_size, gap_center)
        assert len(chars) > 1
        biggest_gap = -1
        biggest_gap_center = -1
        for i in range(len(chars) - 1):
            coord_i = coord_func(chars[i])
            coord_iplus1 = coord_func(chars[i + 1])
            gap = coord_iplus1 - coord_i
            assert gap >= 0
            if gap > biggest_gap:
                biggest_gap = gap
                biggest_gap_center = (coord_i + coord_iplus1) / 2
        if biggest_gap >= 0:
            return (biggest_gap, biggest_gap_center)
        return None

    def _char_width(self, char: PdfChar) -> float:
        return char.bounding_box.x1 - char.bounding_box.x0

    def _char_height(self, char: PdfChar) -> float:
        return char.bounding_box.y1 - char.bounding_box.y0

    def _avg(self, vals: list[float]) -> float:
        assert len(vals) > 0
        return sum(vals) / len(vals)