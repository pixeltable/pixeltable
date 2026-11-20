"""
Pixeltable UDFs for `DocumentType`.
"""

from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, TypedDict, cast

import ftfy
import PIL.Image
from bs4.element import NavigableString, Tag
from pypdfium2 import PdfDocument  # type: ignore[import-untyped]

import pixeltable as pxt
from pixeltable import exceptions as excs, exprs, type_system as ts
from pixeltable.env import Env
from pixeltable.utils.documents import DocumentHandle, get_document_handle
from pixeltable.utils.spacy import get_spacy_model

if TYPE_CHECKING:
    import spacy


class Element(Enum):
    TEXT = 1
    IMAGE = 2


class ChunkMetadata(Enum):
    TITLE = 1
    HEADING = 2
    PAGE = 4


class Separator(Enum):
    HEADING = 1
    PARAGRAPH = 2
    SENTENCE = 3
    TOKEN_LIMIT = 4
    CHAR_LIMIT = 5
    PAGE = 6


class Heading(TypedDict, total=False):
    """The heading hierarchy at a point in a document: the text of each currently open heading level."""

    h1: str
    h2: str
    h3: str
    h4: str
    h5: str
    h6: str


@dataclass
class DocumentSectionMetadata:
    """Metadata for a subsection of a document (ie, a structural element like a heading or paragraph)"""

    # html and markdown metadata
    # the stack of headings up to the most recently observed one;
    # eg, if the most recent one was an h2, 'headings' would contain keys 1 and 2, but nothing below that
    heading: Heading | None = None

    # pdf-specific metadata
    page: int | None = None


@dataclass
class DocumentSection:
    """A single document chunk, according to some of the splitting criteria"""

    text: str | None = None
    image: PIL.Image.Image | None = None
    metadata: DocumentSectionMetadata | None = None


def _parse_separators(separators: str) -> list[Separator]:
    ret: list[Separator] = []
    for s in separators.split(','):
        clean_s = s.strip().upper()
        if not clean_s:
            continue
        if clean_s not in Separator.__members__:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid separator: `{s.strip()}`. Valid separators are: {", ".join(Separator.__members__).lower()}',
            )
        ret.append(Separator[clean_s])
    return ret


def _parse_metadata(metadata: str) -> list[ChunkMetadata]:
    ret: list[ChunkMetadata] = []
    for m in metadata.split(','):
        clean_m = m.strip().upper()
        if not clean_m:
            continue
        if clean_m not in ChunkMetadata.__members__:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid metadata: `{m.strip()}`. Valid metadata are: {", ".join(ChunkMetadata.__members__).lower()}',
            )
        ret.append(ChunkMetadata[clean_m])
    return ret


def _parse_elements(elements: list[Literal['text', 'image']] | None) -> list[Element]:
    if elements is None:
        return [Element.TEXT]

    result: list[Element] = []
    for e in elements:
        clean_e = e.strip().upper()
        if clean_e not in Element.__members__:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid element: `{e}`. Valid elements are: {", ".join(Element.__members__).lower()}',
            )
        result.append(Element[clean_e])
    if len(result) == 0:
        raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'elements cannot be empty')
    return result


_HTML_HEADINGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}


class DocumentChunk(TypedDict, total=False):
    """One chunk of a split document; which fields are present depends on `elements` and `metadata`."""

    text: str
    image: PIL.Image.Image
    title: str | None
    heading: Heading | None
    page: int | None


_METADATA_COLUMN_TYPES = {
    ChunkMetadata.TITLE: ts.String | None,
    ChunkMetadata.HEADING: Heading | None,
    ChunkMetadata.PAGE: ts.Int | None,
}


@pxt.iterator
class document_splitter(pxt.PxtIterator):
    """Iterator over chunks of a document. The document is chunked according to the specified `separators`.

    Chunked text will be cleaned with `ftfy.fix_text` to fix up common problems with unicode sequences.

    __Separators by document format__:

    A separator marked "No effect" is ignored for that format without an error. For example, with
    `separators='page'`, an HTML document in the same table produces a single chunk.

    | Separator | HTML, Markdown, Office | PDF | TXT |
    |---|---|---|---|
    | `'heading'` | Splits at headings | No effect | No effect |
    | `'paragraph'` | Splits at headings and paragraphs | Not supported: the whole `insert()` fails | No effect |
    | `'page'` | No effect | Splits at pages | No effect |
    | `'sentence'` | Splits at headings, paragraphs and sentences | Splits into sentences | Splits into sentences |
    | `'token_limit'`, `'char_limit'` | Splits long chunks | Splits long chunks | Splits long chunks |

    Office documents are PPTX, DOCX and XLSX files. `'token_limit'` and `'char_limit'` split each chunk that is longer
    than `limit` tokens or characters.

    __Outputs__:

        One row per chunk, with the following columns, depending on the specified `elements` and `metadata`:

        - `text` (`pxt.String`): The text of the chunk. Present if `'text'` is specified in `elements`.
        - `image` (`pxt.Image`): The image extracted from the chunk. Present if `'image'` is specified in `elements`.
        - `title` (`pxt.String | None`): The content of the `<title>` element of an HTML document; `''` for other
            formats and for HTML documents without one. Present if `'title'` is specified in `metadata`.
        - `heading` (`Heading | None`): The heading hierarchy at the start of the chunk, with the text of each
            currently open heading level under the keys `h1` through `h6` (HTML, Markdown and Office documents
            only). Present if `'heading'` is specified in `metadata`.
        - `page` (`pxt.Int | None`): The 0-based page number of the chunk. Set only for PDF documents, and only if
            `separators` includes `'page'`; otherwise `None`. Present if `'page'` is specified in `metadata`.

    Args:
        separators: separators to use to chunk the document. Options are:
             `'heading'`, `'paragraph'`, `'sentence'`, `'token_limit'`, `'char_limit'`, `'page'`.
             This may be a comma-separated string, e.g., `'heading,token_limit'`. Which separators apply depends on
             the document format; see the table above. If `separators` includes `'paragraph'`, processing a PDF
             document raises an error that fails the whole `insert()`, or `create_view()` if the base table already
             contains a PDF document.
        elements: list of elements to extract from the document. Options are:
            `'text'`, `'image'`. Defaults to `['text']` if not specified. The `'image'` element is only supported
            for PDF documents, and requires `separators='page'` with no other separators.
        limit: the maximum number of tokens or characters in each chunk, if `'token_limit'`
             or `'char_limit'` is specified.
        overlap: the number of tokens or characters shared by consecutive chunks. Can only be specified
             together with `limit`, and must be smaller than it. Defaults to 0.
        metadata: additional metadata fields to include in the output. Options are:
             `'title'` (HTML), `'heading'` (HTML, Markdown and Office), `'page'` (PDF). The input may be a
             comma-separated string, e.g., `'title,heading'`.
        skip_tags: list of HTML tags to skip when processing HTML documents.
        spacy_model: Name of the spaCy model to use for sentence segmentation. This parameter is ignored unless
            the `'sentence'` separator is specified.
        tiktoken_encoding: Name of the tiktoken encoding to use when counting tokens. This parameter is ignored
            unless the `'token_limit'` separator is specified.
        tiktoken_target_model: Name of the target model to use when counting tokens with tiktoken. If specified,
            this parameter overrides `tiktoken_encoding`. This parameter is ignored unless the `'token_limit'`
            separator is specified.
        image_dpi: DPI to use when extracting images from PDFs. Defaults to 300.
        image_format: format to use when extracting images from PDFs. Defaults to 'png'.

    Examples:
        All these examples assume an existing table `tbl` with a column `doc` of type `pxt.Document`.

        Create a view that splits all documents into chunks of up to 300 tokens:

        >>> pxt.create_view(
        ...     'chunks',
        ...     tbl,
        ...     iterator=document_splitter(
        ...         tbl.doc, separators='token_limit', limit=300
        ...     ),
        ... )

        Create a view that splits all documents along sentence boundaries, including title and heading metadata:

        >>> pxt.create_view(
        ...     'sentence_chunks',
        ...     tbl,
        ...     iterator=document_splitter(
        ...         tbl.doc, separators='sentence', metadata='title,heading'
        ...     ),
        ... )
    """

    _doc_handle: DocumentHandle
    _separators: list[Separator]
    _elements: list[Element]
    _metadata_fields: list[ChunkMetadata]
    _doc_title: str
    _limit: int
    _skip_tags: list[str]
    _overlap: int
    _spacy_model: 'spacy.Language'
    _tiktoken_encoding: str | None
    _tiktoken_target_model: str | None
    _image_dpi: int
    _image_format: str

    _sections: Iterator[DocumentSection]

    def __init__(
        self,
        document: pxt.Document,
        separators: str,
        *,
        elements: list[Literal['text', 'image']] | None = None,
        limit: int | None = None,
        overlap: int | None = None,
        metadata: str = '',
        skip_tags: list[str] | None = None,
        spacy_model: str = 'en_core_web_sm',
        tiktoken_encoding: str | None = 'cl100k_base',
        tiktoken_target_model: str | None = None,
        image_dpi: int = 300,
        image_format: str = 'png',
    ) -> None:
        if skip_tags is None:
            skip_tags = ['nav']
        self._doc_handle = get_document_handle(document)
        self._elements = _parse_elements(elements)
        assert self._doc_handle is not None
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
        self._skip_tags = skip_tags
        self._overlap = 0 if overlap is None else overlap
        self._tiktoken_encoding = tiktoken_encoding
        self._tiktoken_target_model = tiktoken_target_model
        self._image_dpi = image_dpi
        self._image_format = image_format

        # set up processing pipeline
        if self._doc_handle.format == ts.DocumentType.DocumentFormat.HTML:
            assert self._doc_handle.bs_doc is not None
            self._sections = self._html_sections()
        elif self._doc_handle.format == ts.DocumentType.DocumentFormat.MD:
            assert self._doc_handle.md_ast is not None
            self._sections = self._markdown_sections(self._doc_handle.md_ast)
        elif self._doc_handle.format == ts.DocumentType.DocumentFormat.PDF:
            assert self._doc_handle.pdf_doc is not None
            self._sections = self._pdf_sections()
        elif self._doc_handle.format == ts.DocumentType.DocumentFormat.TXT:
            assert self._doc_handle.txt_doc is not None
            self._sections = self._txt_sections()
        elif self._doc_handle.format in (
            ts.DocumentType.DocumentFormat.PPTX,
            ts.DocumentType.DocumentFormat.DOCX,
            ts.DocumentType.DocumentFormat.XLSX,
        ):
            assert self._doc_handle.md_ast is not None
            self._sections = self._markdown_sections(self._doc_handle.md_ast)
        else:
            raise AssertionError(f'Unsupported document format: {self._doc_handle.format}')

        if Separator.SENTENCE in self._separators:
            self._sections = self._sentence_sections(self._sections)
            self._spacy_model = get_spacy_model(spacy_model)
        if Separator.TOKEN_LIMIT in self._separators:
            self._sections = self._token_chunks(self._sections)
        if Separator.CHAR_LIMIT in self._separators:
            self._sections = self._char_chunks(self._sections)

    def __next__(self) -> DocumentChunk:
        while True:
            section = next(self._sections)
            if section.text is None and section.image is None:
                continue
            result: DocumentChunk = {}
            for element in self._elements:
                if element == Element.TEXT:
                    result['text'] = section.text
                elif element == Element.IMAGE:
                    result['image'] = section.image

            for md_field in self._metadata_fields:
                if md_field == ChunkMetadata.TITLE:
                    result['title'] = self._doc_title
                elif md_field == ChunkMetadata.HEADING:
                    result['heading'] = section.metadata.heading
                elif md_field == ChunkMetadata.PAGE:
                    result['page'] = section.metadata.page

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

        def update_metadata(el: bs4.Tag) -> None:
            # update current state
            nonlocal headings
            if el.name in _HTML_HEADINGS:
                # remove the previously seen lower levels
                lower_levels = [lv for lv in headings if lv > el.name]
                for lv in lower_levels:
                    del headings[lv]
                headings[el.name] = el.get_text().strip()

        def emit() -> Iterator[DocumentSection]:
            nonlocal accumulated_text, headings
            if len(accumulated_text) > 0:
                md = DocumentSectionMetadata(heading=cast(Heading, headings.copy()))
                full_text = ' '.join(accumulated_text)
                full_text = ftfy.fix_text(full_text)
                yield DocumentSection(text=full_text, metadata=md)
                accumulated_text = []

        def process_element(el: Tag | NavigableString) -> Iterator[DocumentSection]:
            # process the element and emit sections as necessary
            nonlocal accumulated_text, headings, emit_on_heading, emit_on_paragraph

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
            for child in el.children:
                assert isinstance(child, (bs4.element.Tag, bs4.NavigableString)), type(el)
                yield from process_element(child)

        yield from process_element(self._doc_handle.bs_doc)
        yield from emit()

    def _markdown_sections(self, md_ast: list[dict]) -> Iterator[DocumentSection]:
        """Create DocumentSections from a markdown AST."""
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
                metadata = DocumentSectionMetadata(heading=cast(Heading, headings.copy()))
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

        for el in md_ast:
            yield from process_element(el)
        yield from emit()

    def _pdf_sections(self) -> Iterator[DocumentSection]:
        if Separator.PARAGRAPH in self._separators:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                "Separator 'paragraph' is not supported for PDF documents; "
                "use 'page', 'sentence', 'token_limit' or 'char_limit'",
            )

        doc: PdfDocument = self._doc_handle.pdf_doc
        assert isinstance(doc, PdfDocument)

        emit_on_page = Separator.PAGE in self._separators
        accumulated_text: list[str] = []

        def _add_cleaned(raw: str) -> None:
            fixed = ftfy.fix_text(raw)
            if fixed:
                accumulated_text.append(fixed)

        def _emit_text() -> str:
            txt = ''.join(accumulated_text)
            accumulated_text.clear()
            return txt

        for page_idx, page in enumerate(doc):
            with closing(page):
                img: PIL.Image.Image | None = None
                if Element.IMAGE in self._elements:
                    with closing(page.render()) as bitmap:
                        img = bitmap.to_pil().copy()
                with closing(page.get_textpage()) as textpage:
                    text = textpage.get_text_bounded()
            _add_cleaned(text)
            if emit_on_page and (accumulated_text or img is not None):
                md = DocumentSectionMetadata(page=page_idx)
                yield DocumentSection(text=_emit_text(), image=img, metadata=md)

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
                doc = self._spacy_model(section.text)
                for sent in doc.sents:
                    yield DocumentSection(text=sent.text, metadata=section.metadata)

    def _token_chunks(self, input: Iterable[DocumentSection]) -> Iterator[DocumentSection]:
        import tiktoken

        if self._tiktoken_target_model is not None:
            encoding = tiktoken.encoding_for_model(self._tiktoken_target_model)
        else:
            encoding = tiktoken.get_encoding(self._tiktoken_encoding)
        assert self._overlap >= 0, self._overlap
        assert self._overlap < self._limit, (self._overlap, self._limit)

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

                # If we couldn't find a valid decode point, force progress by moving forward
                if end_idx <= start_idx:
                    # Try to decode with replacement errors to make progress
                    end_idx = min(start_idx + self._limit, len(tokens))
                    text = encoding.decode(tokens[start_idx:end_idx], errors='replace')

                assert end_idx > start_idx
                assert text is not None
                yield DocumentSection(text=text, metadata=section.metadata)
                if end_idx == len(tokens):
                    break
                start_idx = max(start_idx + 1, end_idx - self._overlap)  # ensure we make progress

    def _char_chunks(self, input: Iterable[DocumentSection]) -> Iterator[DocumentSection]:
        assert self._overlap >= 0, self._overlap
        assert self._overlap < self._limit, (self._overlap, self._limit)
        for section in input:
            if section.text is None:
                continue
            start_idx = 0
            while start_idx < len(section.text):
                end_idx = min(start_idx + self._limit, len(section.text))
                text = section.text[start_idx:end_idx]
                yield DocumentSection(text=text, metadata=section.metadata)
                if end_idx == len(section.text):
                    break
                start_idx += self._limit - self._overlap

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        elements = _parse_elements(bound_args.get('elements'))
        _parse_metadata(bound_args.get('metadata', ''))

        assert 'separators' in bound_args
        separators = _parse_separators(bound_args['separators'])

        limit = bound_args.get('limit')
        overlap = bound_args.get('overlap')

        if Element.IMAGE in elements and separators != [Separator.PAGE]:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                "elements=['image'] requires separators='page' with no other separators",
            )
        if Separator.TOKEN_LIMIT in separators or Separator.CHAR_LIMIT in separators:
            if Separator.TOKEN_LIMIT in separators and Separator.CHAR_LIMIT in separators:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    "Cannot specify both 'token_limit' and 'char_limit' separators",
                )
            if limit is None:
                raise excs.RequestError(
                    excs.ErrorCode.MISSING_REQUIRED,
                    "'limit' must be specified with 'token_limit' and 'char_limit' separators",
                )
            if limit <= 0:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, "'limit' must be a positive integer")
            if overlap is not None and overlap < 0:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, "'overlap' must be a non-negative integer")
            if overlap is not None and overlap >= limit:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, "'overlap' must be smaller than 'limit'")
        elif limit is not None or overlap is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                "limit/overlap requires the 'token_limit' or 'char_limit' separator",
            )

        if Separator.SENTENCE in separators:
            # Validate spaCy model
            _ = get_spacy_model(bound_args.get('spacy_model', 'en_core_web_sm'))

        if Separator.TOKEN_LIMIT in separators:
            Env.get().require_package('tiktoken')

    @classmethod
    def conditional_output_schema(cls, bound_args: dict[str, exprs.Expr]) -> dict[str, type]:
        schema: dict[str, type] = {}

        elements = [Element.TEXT]
        if bound_args.get('elements') is not None:
            if not isinstance(bound_args['elements'], exprs.Literal):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, 'document_splitter(): `elements` must be a constant expression.'
                )
            elements = _parse_elements(bound_args['elements'].val)
        for element in elements:
            if element == Element.TEXT:
                schema['text'] = ts.String
            elif element == Element.IMAGE:
                schema['image'] = ts.Image

        if bound_args.get('metadata') is not None:
            if not isinstance(bound_args['metadata'], exprs.Literal):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, 'document_splitter(): `metadata` must be a constant expression.'
                )
            md_fields = _parse_metadata(bound_args['metadata'].val)
            for md_field in md_fields:
                schema[md_field.name.lower()] = _METADATA_COLUMN_TYPES[md_field]  # type: ignore[assignment]

        return schema

    def close(self) -> None:
        self._doc_handle.close()
