from typing import Dict, Any, List, Tuple, Generator, Optional, Iterable
import logging
import dataclasses
import enum

from .base import ComponentIterator

from pixeltable.type_system import ColumnType, DocumentType, StringType, IntType, JsonType
from pixeltable.exceptions import Error
from pixeltable.env import Env


_logger = logging.getLogger('pixeltable')


class HTMLMetadata(enum.Enum):
    TITLE = 1
    HEADINGS = 2
    SOURCELINE = 3


class HTMLSectionType(enum.Enum):
    HEADING = 1
    PARAGRAPH = 2
    TEXT = 3  # TODO: remove this, since it's not really a structural element?


class Separator(enum.Enum):
    HEADING = 1
    PARAGRAPH = 2
    SENTENCE = 3
    TOKEN_LIMIT = 4
    CHAR_LIMIT = 5


@dataclasses.dataclass
class HTMLSectionMd:
    """Metadata for a subsection of an html document (ie, a structural element like a heading or paragraph)"""
    source_line: int

    # the stack of headings up to the most recently observed one;
    # eg, if the most recent one was an h2, 'headings' would contain keys 1 and 2, but nothing below that
    headings: Dict[int, str]


@dataclasses.dataclass
class DocumentSection:
    """A single document chunk, according to some of the splitting criteria"""
    text: Optional[str]
    md: Optional[HTMLSectionMd]


class HTMLDocumentSplitter(ComponentIterator):
    """"Iterator over sections of an HTML document
    """
    MD_COLUMN_TYPES = {
        HTMLMetadata.TITLE: StringType(),
        HTMLMetadata.HEADINGS: JsonType(),
        HTMLMetadata.SOURCELINE: IntType()
    }

    def __init__(
            self, document: str, *, separators: str, limit: int = 0, overlap: int = 0, metadata: str = '',
            skip_tags: List[str] = ['nav'], tiktoken_encoding: Optional[str] = 'cl100k_base',
            tiktoken_target_model: Optional[str] = None
    ):
        import bs4
        with open(document, 'r') as fh:
            self._bs_doc = bs4.BeautifulSoup(fh, 'html.parser')
        self._separators = [Separator[s.upper()] for s in separators.split(',')]
        self._md_fields = [HTMLMetadata[m.upper()] for m in metadata.split(',')]
        self._doc_title = self._bs_doc.title.get_text(strip=True)
        self._limit = limit
        self._skip_tags = skip_tags
        self._overlap = overlap
        self._tiktoken_encoding = tiktoken_encoding
        self._tiktoken_target_model = tiktoken_target_model

        # set up processing pipeline
        self._sections = self._html_sections()  # we always start with html sections
        if Separator.SENTENCE in self._separators:
            self._sections = self._sentence_sections(self._sections)
        if Separator.TOKEN_LIMIT in self._separators:
            self._sections = self._token_chunks(self._sections)
        if Separator.CHAR_LIMIT in self._separators:
            self._sections = self._char_chunks(self._sections)

    @classmethod
    def input_schema(cls) -> Dict[str, ColumnType]:
        return {
            'document': DocumentType(nullable=False),
            'separators': StringType(nullable=False),
            'metadata': StringType(nullable=True),
            'limit': IntType(nullable=True),
            'overlap': IntType(nullable=True),
            'skip_tags': StringType(nullable=True),
            'tiktoken_encoding': StringType(nullable=True),
            'tiktoken_target_model': StringType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> Tuple[Dict[str, ColumnType], List[str]]:
        schema = {'text': StringType()}
        if 'metadata' in kwargs:
            md_fields = kwargs['metadata'].split(',')
            for md_field in md_fields:
                if not hasattr(HTMLMetadata, md_field.upper()):
                    raise Error(f'Invalid metadata field {md_field}')
                schema[md_field.lower()] = cls.MD_COLUMN_TYPES[HTMLMetadata[md_field.upper()]]

        assert 'separators' in kwargs
        separators = kwargs['separators'].split(',')
        for separator in separators:
            if not hasattr(Separator, separator.upper()):
                raise Error(f'Invalid separator {separator}')

        # check dependencies
        if 'sentence' in separators:
            Env.get().require_package('spacy')
        if 'token_limit' in separators:
            Env.get().require_package('tiktoken')

        if 'limit' in kwargs or 'overlap' in kwargs:
            if 'token_limit' not in separators and 'char_limit' not in separators:
                raise Error('limit/overlap requires the "token_limit" or "char_limit" separator')
            if 'limit' in kwargs and int(kwargs['limit']) <= 0:
                raise Error('"limit" must be an integer > 0')
            if 'overlap' in kwargs and int(kwargs['overlap']) < 0:
                raise Error('"overlap" must be an integer >= 0')
        if 'token_limit' in separators or 'char_limit' in separators:
            if 'token_limit' in separators and 'char_limit' in separators:
                raise Error('Cannot specify both "token_limit" and "char_limit" separators')
            if 'limit' not in kwargs:
                raise Error('limit is required with "token_limit"/"char_limit" separators')

        return schema, []

    def __next__(self) -> Dict[str, Any]:
        while True:
            section = next(self._sections)
            if section.text is None:
                continue
            result = {'text': section.text}
            for md_field in self._md_fields:
                if md_field == HTMLMetadata.TITLE:
                    result[md_field.name.lower()] = self._doc_title
                elif md_field == HTMLMetadata.HEADINGS:
                    result[md_field.name.lower()] = section.md.headings
                elif md_field == HTMLMetadata.SOURCELINE:
                    result[md_field.name.lower()] = section.md.source_line
            return result

    def _html_sections(self) -> Generator[DocumentSection, None, None]:
        """Create DocumentSections reflecting the html-specific separators"""
        import bs4
        emit_on_paragraph = Separator.PARAGRAPH in self._separators or Separator.SENTENCE in self._separators
        emit_on_heading = Separator.HEADING in self._separators or emit_on_paragraph
        # current state
        text_section = ''  # currently accumulated text
        headings: Dict[int, str] = {}   # current state of observed headings (level -> text)
        sourceline = 0  # most recently seen sourceline

        def update_md(el: bs4.Tag) -> None:
            # update current state
            nonlocal headings, sourceline
            sourceline = el.sourceline
            if el.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(el.name[1])
                # remove the previously seen lower levels
                lower_levels = [l for l in headings.keys() if l > level]
                for l in lower_levels:
                    del headings[l]
                headings[level] = el.get_text(strip=True)

        def emit() -> None:
            nonlocal text_section, headings, sourceline
            if len(text_section) > 0:
                md = HTMLSectionMd(sourceline, headings.copy())
                yield DocumentSection(text=text_section, md=md)
                text_section = ''

        def process_element(el: bs4.PageElement) -> Generator[DocumentSection, None, None]:
            # process the element and emit sections as necessary
            nonlocal text_section, headings, sourceline, emit_on_heading, emit_on_paragraph
            if el.name in self._skip_tags:
                return

            if isinstance(el, bs4.NavigableString):
                # accumulate text until we see a tag we care about
                text = el.get_text().strip()
                if len(text) > 0:
                    text_section += ' ' + text
                return

            if el.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if emit_on_heading:
                    yield from emit()
                update_md(el)
            elif el.name == 'p':
                if emit_on_paragraph:
                    yield from emit()
                update_md(el)
            for child in el.children:
                yield from process_element(child)

        yield from process_element(self._bs_doc)
        yield from emit()

    def _sentence_sections(self, input_sections: Iterable[DocumentSection]) -> Generator[DocumentSection, None, None]:
        """Split the input sections into sentences"""
        for section in input_sections:
            if section.text is not None:
                doc = Env.get().spacy_nlp(section.text)
                for sent in doc.sents:
                    yield DocumentSection(text=sent.text, md=section.md)

    def _token_chunks(self, input: Iterable[DocumentSection]) -> Generator[DocumentSection, None, None]:
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
            while start_idx < len(tokens):
                end_idx = min(start_idx + self._limit, len(tokens))
                text = encoding.decode(tokens[start_idx:end_idx])
                yield DocumentSection(text=text, md=section.md)
                start_idx += self._limit - self._overlap

    def _char_chunks(self, input: Iterable[DocumentSection]) -> Generator[DocumentSection, None, None]:
        for section in input:
            if section.text is None:
                continue
            start_idx = 0
            while start_idx < len(section.text):
                end_idx = min(start_idx + self._limit, len(section.text))
                text = section.text[start_idx:end_idx]
                yield DocumentSection(text=text, md=section.md)
                start_idx += self._limit - self._overlap

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        pass
