import dataclasses
import enum
import logging
from typing import Any, ClassVar, Iterable, Iterator, Optional, Union

import ftfy

from pixeltable.env import Env
from pixeltable.exceptions import Error
from pixeltable.type_system import ColumnType, DocumentType, IntType, JsonType, StringType
from pixeltable.utils.documents import get_document_handle

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class ChunkMetadata(enum.Enum):
    TITLE = 1
    HEADING = 2
    SOURCELINE = 3
    PAGE = 4
    BOUNDING_BOX = 5


class Separator(enum.Enum):
    HEADING = 1
    PARAGRAPH = 2
    SENTENCE = 3
    TOKEN_LIMIT = 4
    CHAR_LIMIT = 5
    PAGE = 6


@dataclasses.dataclass
class DocumentSectionMetadata:
    """Metadata for a subsection of a document (ie, a structural element like a heading or paragraph)"""

    # html and markdown metadata
    sourceline: Optional[int] = None
    # the stack of headings up to the most recently observed one;
    # eg, if the most recent one was an h2, 'headings' would contain keys 1 and 2, but nothing below that
    heading: Optional[dict[str, str]] = None

    # pdf-specific metadata
    page: Optional[int] = None
    # bounding box as an {x1, y1, x2, y2} dictionary
    bounding_box: Optional[dict[str, float]] = None


@dataclasses.dataclass
class DocumentSection:
    """A single document chunk, according to some of the splitting criteria"""

    text: Optional[str]
    metadata: Optional[DocumentSectionMetadata]


def _parse_separators(separators: str) -> list[Separator]:
    ret = []
    for s in separators.split(','):
        clean_s = s.strip().upper()
        if not clean_s:
            continue
        if clean_s not in Separator.__members__:
            raise Error(
                f'Invalid separator: `{s.strip()}`. Valid separators are: {", ".join(Separator.__members__).lower()}'
            )
        ret.append(Separator[clean_s])
    return ret


def _parse_metadata(metadata: str) -> list[ChunkMetadata]:
    ret = []
    for m in metadata.split(','):
        clean_m = m.strip().upper()
        if not clean_m:
            continue
        if clean_m not in ChunkMetadata.__members__:
            raise Error(
                f'Invalid metadata: `{m.strip()}`. Valid metadata are: {", ".join(ChunkMetadata.__members__).lower()}'
            )
        ret.append(ChunkMetadata[clean_m])
    return ret


_HTML_HEADINGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}


class DocumentSplitter(ComponentIterator):
    """Iterator over chunks of a document. The document is chunked according to the specified `separators`.

    The iterator yields a `text` field containing the text of the chunk, and it may also
    include additional metadata fields if specified in the `metadata` parameter, as explained below.

    Chunked text will be cleaned with `ftfy.fix_text` to fix up common problems with unicode sequences.
    """

    METADATA_COLUMN_TYPES: ClassVar[dict[ChunkMetadata, ColumnType]] = {
        ChunkMetadata.TITLE: StringType(nullable=True),
        ChunkMetadata.HEADING: JsonType(nullable=True),
        ChunkMetadata.SOURCELINE: IntType(nullable=True),
        ChunkMetadata.PAGE: IntType(nullable=True),
        ChunkMetadata.BOUNDING_BOX: JsonType(nullable=True),
    }

    def __init__(
        self,
        document: str,
        *,
        separators: str,
        limit: Optional[int] = None,
        overlap: Optional[int] = None,
        metadata: str = '',
        html_skip_tags: Optional[list[str]] = None,
        tiktoken_encoding: Optional[str] = 'cl100k_base',
        tiktoken_target_model: Optional[str] = None,
    ):
        """Init method for `DocumentSplitter` class.

        Args:
            separators: separators to use to chunk the document. Options are:
                 `'heading'`, `'paragraph'`, `'sentence'`, `'token_limit'`, `'char_limit'`, `'page'`.
                 This may be a comma-separated string, e.g., `'heading,token_limit'`.
            limit: the maximum number of tokens or characters in each chunk, if `'token_limit'`
                 or `'char_limit'` is specified.
            metadata: additional metadata fields to include in the output. Options are:
                 `'title'`, `'heading'` (HTML and Markdown), `'sourceline'` (HTML), `'page'` (PDF), `'bounding_box'`
                 (PDF). The input may be a comma-separated string, e.g., `'title,heading,sourceline'`.
        """
        if html_skip_tags is None:
            html_skip_tags = ['nav']
        self._doc_handle = get_document_handle(document)
        assert self._doc_handle is not None
        # calling the output_schema method to validate the input arguments
        self.output_schema(separators=separators, metadata=metadata, limit=limit, overlap=overlap)
        self._separators = _parse_separators(separators)
        self._metadata_fields = _parse_metadata(metadata)
        if self._doc_handle.bs_doc is not None:
            title = self._doc_handle.bs_doc.title
            if title is None:
                self._doc_title = ''
            else:
                self._doc_title = ftfy.fix_text(title.get_text().strip())
        else:
            self._doc_title = ''
        self._limit = 0 if limit is None else limit
        self._skip_tags = html_skip_tags
        self._overlap = 0 if overlap is None else overlap
        self._tiktoken_encoding = tiktoken_encoding
        self._tiktoken_target_model = tiktoken_target_model

        # set up processing pipeline
        if self._doc_handle.format == DocumentType.DocumentFormat.HTML:
            assert self._doc_handle.bs_doc is not None
            self._sections = self._html_sections()
        elif self._doc_handle.format == DocumentType.DocumentFormat.MD:
            assert self._doc_handle.md_ast is not None
            self._sections = self._markdown_sections()
        elif self._doc_handle.format == DocumentType.DocumentFormat.PDF:
            assert self._doc_handle.pdf_doc is not None
            self._sections = self._pdf_sections()
        elif self._doc_handle.format == DocumentType.DocumentFormat.TXT:
            assert self._doc_handle.txt_doc is not None
            self._sections = self._txt_sections()
        else:
            raise AssertionError(f'Unsupported document format: {self._doc_handle.format}')

        if Separator.SENTENCE in self._separators:
            self._sections = self._sentence_sections(self._sections)
        if Separator.TOKEN_LIMIT in self._separators:
            self._sections = self._token_chunks(self._sections)
        if Separator.CHAR_LIMIT in self._separators:
            self._sections = self._char_chunks(self._sections)

    @classmethod
    def input_schema(cls) -> dict[str, ColumnType]:
        return {
            'document': DocumentType(nullable=False),
            'separators': StringType(nullable=False),
            'metadata': StringType(nullable=False),
            'limit': IntType(nullable=True),
            'overlap': IntType(nullable=True),
            'skip_tags': StringType(nullable=True),
            'tiktoken_encoding': StringType(nullable=True),
            'tiktoken_target_model': StringType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ColumnType], list[str]]:
        schema: dict[str, ColumnType] = {'text': StringType()}
        md_fields = _parse_metadata(kwargs['metadata']) if 'metadata' in kwargs else []

        for md_field in md_fields:
            schema[md_field.name.lower()] = cls.METADATA_COLUMN_TYPES[md_field]

        assert 'separators' in kwargs
        separators = _parse_separators(kwargs['separators'])

        limit = kwargs.get('limit')
        overlap = kwargs.get('overlap')

        if limit is not None or overlap is not None:
            if Separator.TOKEN_LIMIT not in separators and Separator.CHAR_LIMIT not in separators:
                raise Error('limit/overlap requires the "token_limit" or "char_limit" separator')
            if limit is not None and limit <= 0:
                raise Error('"limit" must be an integer > 0')
            if overlap is not None and overlap < 0:
                raise Error('"overlap" must be an integer >= 0')
        if Separator.TOKEN_LIMIT in separators or Separator.CHAR_LIMIT in separators:
            if Separator.TOKEN_LIMIT in separators and Separator.CHAR_LIMIT in separators:
                raise Error('Cannot specify both "token_limit" and "char_limit" separators')
            if kwargs.get('limit') is None:
                raise Error('limit is required with "token_limit"/"char_limit" separators')

        # check dependencies at the end
        if Separator.SENTENCE in separators:
            _ = Env.get().spacy_nlp
        if Separator.TOKEN_LIMIT in separators:
            Env.get().require_package('tiktoken')

        return schema, []

    def __next__(self) -> dict[str, Any]:
        while True:
            section = next(self._sections)
            if section.text is None:
                continue
            result: dict[str, Any] = {'text': section.text}
            for md_field in self._metadata_fields:
                if md_field == ChunkMetadata.TITLE:
                    result[md_field.name.lower()] = self._doc_title
                elif md_field == ChunkMetadata.HEADING:
                    result[md_field.name.lower()] = section.metadata.heading
                elif md_field == ChunkMetadata.SOURCELINE:
                    result[md_field.name.lower()] = section.metadata.sourceline
                elif md_field == ChunkMetadata.PAGE:
                    result[md_field.name.lower()] = section.metadata.page
                elif md_field == ChunkMetadata.BOUNDING_BOX:
                    result[md_field.name.lower()] = section.metadata.bounding_box
            return result

    def _html_sections(self) -> Iterator[DocumentSection]:
        """Create DocumentSections reflecting the html-specific separators"""
        import bs4

        emit_on_paragraph = Separator.PARAGRAPH in self._separators or Separator.SENTENCE in self._separators
        emit_on_heading = Separator.HEADING in self._separators or emit_on_paragraph
        # current state
        accumulated_text: list[str] = []  # currently accumulated text
        # accumulate pieces then join before emit to avoid quadratic complexity of string concatenation

        headings: dict[str, str] = {}  # current state of observed headings (level -> text)
        sourceline = 0  # most recently seen sourceline

        def update_metadata(el: bs4.Tag) -> None:
            # update current state
            nonlocal headings, sourceline
            sourceline = el.sourceline
            if el.name in _HTML_HEADINGS:
                # remove the previously seen lower levels
                lower_levels = [lv for lv in headings if lv > el.name]
                for lv in lower_levels:
                    del headings[lv]
                headings[el.name] = el.get_text().strip()

        def emit() -> Iterator[DocumentSection]:
            nonlocal accumulated_text, headings, sourceline
            if len(accumulated_text) > 0:
                md = DocumentSectionMetadata(sourceline=sourceline, heading=headings.copy())
                full_text = ' '.join(accumulated_text)
                full_text = ftfy.fix_text(full_text)
                yield DocumentSection(text=full_text, metadata=md)
                accumulated_text = []

        def process_element(el: Union[bs4.element.Tag, bs4.NavigableString]) -> Iterator[DocumentSection]:
            # process the element and emit sections as necessary
            nonlocal accumulated_text, headings, sourceline, emit_on_heading, emit_on_paragraph

            if el.name in self._skip_tags:
                return

            if isinstance(el, bs4.NavigableString):
                # accumulate text until we see a tag we care about
                text = el.get_text().strip()
                if len(text) > 0:
                    accumulated_text.append(text)
                return

            if el.name in _HTML_HEADINGS:
                if emit_on_heading:
                    yield from emit()
                update_metadata(el)
            elif el.name == 'p':
                if emit_on_paragraph:
                    yield from emit()
                update_metadata(el)
            for child in el.children:
                assert isinstance(child, (bs4.element.Tag, bs4.NavigableString)), type(el)
                yield from process_element(child)

        yield from process_element(self._doc_handle.bs_doc)
        yield from emit()

    def _markdown_sections(self) -> Iterator[DocumentSection]:
        """Create DocumentSections reflecting the html-specific separators"""
        assert self._doc_handle.md_ast is not None
        emit_on_paragraph = Separator.PARAGRAPH in self._separators or Separator.SENTENCE in self._separators
        emit_on_heading = Separator.HEADING in self._separators or emit_on_paragraph
        # current state
        accumulated_text: list[str] = []  # currently accumulated text
        # accumulate pieces then join before emit to avoid quadratic complexity of string concatenation
        headings: dict[str, str] = {}  # current state of observed headings (level -> text)

        def update_headings(heading: dict) -> None:
            # update current state
            nonlocal headings
            assert 'type' in heading and heading['type'] == 'heading'
            lint = heading['attrs']['level']
            level = f'h{lint}'
            text = heading['children'][0]['raw'].strip()
            # remove the previously seen lower levels
            lower_levels = [lv for lv in headings if lv > level]
            for lv in lower_levels:
                del headings[lv]
            headings[level] = text

        def emit() -> Iterator[DocumentSection]:
            nonlocal accumulated_text, headings
            if len(accumulated_text) > 0:
                metadata = DocumentSectionMetadata(sourceline=0, heading=headings.copy())
                yield DocumentSection(text=ftfy.fix_text(' '.join(accumulated_text)), metadata=metadata)
                accumulated_text = []

        def process_element(el: dict) -> Iterator[DocumentSection]:
            # process the element and emit sections as necessary
            nonlocal accumulated_text, headings, emit_on_heading, emit_on_paragraph
            assert 'type' in el

            if el['type'] == 'text':
                # accumulate text until we see a separator element
                text = el['raw'].strip()
                if len(text) > 0:
                    accumulated_text.append(text)
                return

            if el['type'] == 'heading':
                if emit_on_heading:
                    yield from emit()
                update_headings(el)
            elif el['type'] == 'paragraph':
                if emit_on_paragraph:
                    yield from emit()
            if 'children' not in el:
                return
            for child in el['children']:
                yield from process_element(child)

        for el in self._doc_handle.md_ast:
            yield from process_element(el)
        yield from emit()

    def _pdf_sections(self) -> Iterator[DocumentSection]:
        """Create DocumentSections reflecting the pdf-specific separators"""
        import fitz  # type: ignore[import-untyped]

        doc: fitz.Document = self._doc_handle.pdf_doc
        assert doc is not None

        emit_on_paragraph = Separator.PARAGRAPH in self._separators or Separator.SENTENCE in self._separators
        emit_on_page = Separator.PAGE in self._separators or emit_on_paragraph

        accumulated_text = []  # invariant: all elements are ftfy clean and non-empty

        def _add_cleaned_text(raw_text: str) -> None:
            fixed = ftfy.fix_text(raw_text)
            if fixed:
                accumulated_text.append(fixed)

        def _emit_text() -> str:
            full_text = ''.join(accumulated_text)
            accumulated_text.clear()
            return full_text

        for page_number, page in enumerate(doc.pages()):
            for block in page.get_text('blocks'):
                # there is no concept of paragraph in pdf, block is the closest thing
                # we can get (eg a paragraph in text may cut across pages)
                # see pymupdf docs https://pymupdf.readthedocs.io/en/latest/app1.html
                # other libraries like pdfminer also lack an explicit paragraph concept
                x1, y1, x2, y2, text, _, _ = block
                _add_cleaned_text(text)
                if accumulated_text and emit_on_paragraph:
                    bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    metadata = DocumentSectionMetadata(page=page_number, bounding_box=bbox)
                    yield DocumentSection(text=_emit_text(), metadata=metadata)

            if accumulated_text and emit_on_page and not emit_on_paragraph:
                yield DocumentSection(text=_emit_text(), metadata=DocumentSectionMetadata(page=page_number))
                accumulated_text = []

        if accumulated_text and not emit_on_page:
            yield DocumentSection(text=_emit_text(), metadata=DocumentSectionMetadata())

    def _txt_sections(self) -> Iterator[DocumentSection]:
        """Create DocumentSections for text files.

        Currently, it returns the entire text as a single section.
        TODO: Add support for paragraphs.
        """
        assert self._doc_handle.txt_doc is not None
        yield DocumentSection(text=ftfy.fix_text(self._doc_handle.txt_doc), metadata=DocumentSectionMetadata())

    def _sentence_sections(self, input_sections: Iterable[DocumentSection]) -> Iterator[DocumentSection]:
        """Split the input sections into sentences"""
        for section in input_sections:
            if section.text is not None:
                doc = Env.get().spacy_nlp(section.text)
                for sent in doc.sents:
                    yield DocumentSection(text=sent.text, metadata=section.metadata)

    def _token_chunks(self, input: Iterable[DocumentSection]) -> Iterator[DocumentSection]:
        import tiktoken

        if self._tiktoken_target_model is not None:
            encoding = tiktoken.encoding_for_model(self._tiktoken_target_model)
        else:
            encoding = tiktoken.get_encoding(self._tiktoken_encoding)
        assert self._limit > 0 and self._overlap >= 0

        for section in input:
            if section.text is None:
                continue
            tokens = encoding.encode(section.text)
            start_idx = 0
            text = None
            while start_idx < len(tokens):
                end_idx = min(start_idx + self._limit, len(tokens))
                while end_idx > start_idx:
                    # find a cutoff point that doesn't cut in the middle of utf8 multi-byte sequences
                    try:
                        # check that the truncated data can be properly decoded
                        text = encoding.decode(tokens[start_idx:end_idx], errors='strict')
                        break
                    except UnicodeDecodeError:
                        # we split the token array at a point where the utf8 encoding is broken
                        end_idx -= 1

                assert end_idx > start_idx
                assert text
                yield DocumentSection(text=text, metadata=section.metadata)
                start_idx = max(start_idx + 1, end_idx - self._overlap)  # ensure we make progress

    def _char_chunks(self, input: Iterable[DocumentSection]) -> Iterator[DocumentSection]:
        for section in input:
            if section.text is None:
                continue
            start_idx = 0
            while start_idx < len(section.text):
                end_idx = min(start_idx + self._limit, len(section.text))
                text = section.text[start_idx:end_idx]
                yield DocumentSection(text=text, metadata=section.metadata)
                start_idx += self._limit - self._overlap

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        pass
