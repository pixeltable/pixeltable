import dataclasses
import os

import bs4
import puremagic
from pypdfium2 import PdfDocument, PdfiumError  # type: ignore[import-untyped]

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: bs4.BeautifulSoup | None = None
    md_ast: list | None = None
    pdf_doc: PdfDocument | None = None
    txt_doc: str | None = None

    def close(self) -> None:
        # pdf_doc keeps the source file open until closed; the other handles are parsed into memory
        if self.pdf_doc is not None:
            self.pdf_doc.close()

    @classmethod
    def validate(cls, path: str) -> None:
        # open the document to confirm it is readable, then release the file handle
        get_document_handle(path).close()


def get_document_handle(path: str) -> DocumentHandle:
    try:
        _, extension = os.path.splitext(path)
        handle = get_handle_by_extension(path, extension)
        if handle is not None:
            return handle

        # if no extension, use puremagic to determine the type
        extension = puremagic.from_file(path)
        handle = get_handle_by_extension(path, extension)
        if handle is not None:
            return handle
    except OSError as exc:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_DATA_FORMAT, f'Cannot read document {path}: {exc.strerror or exc}'
        ) from exc

    raise excs.RequestError(excs.ErrorCode.INVALID_DATA_FORMAT, f'Unrecognized document format: {path}')


def get_handle_by_extension(path: str, extension: str) -> DocumentHandle | None:
    doc_format = ts.DocumentType.DocumentFormat.from_extension(extension)

    if doc_format == ts.DocumentType.DocumentFormat.HTML:
        return DocumentHandle(doc_format, bs_doc=get_html_handle(path))
    if doc_format == ts.DocumentType.DocumentFormat.MD:
        return DocumentHandle(doc_format, md_ast=get_markdown_handle(path))
    if doc_format == ts.DocumentType.DocumentFormat.PDF:
        return DocumentHandle(doc_format, pdf_doc=get_pdf_handle(path))
    if doc_format == ts.DocumentType.DocumentFormat.XML:
        return DocumentHandle(doc_format, bs_doc=get_xml_handle(path))
    if doc_format == ts.DocumentType.DocumentFormat.TXT:
        return DocumentHandle(doc_format, txt_doc=get_txt(path))
    if doc_format == ts.DocumentType.DocumentFormat.PPTX:
        return DocumentHandle(doc_format, md_ast=get_office_handle(path))
    if doc_format == ts.DocumentType.DocumentFormat.DOCX:
        return DocumentHandle(doc_format, md_ast=get_office_handle(path))
    if doc_format == ts.DocumentType.DocumentFormat.XLSX:
        return DocumentHandle(doc_format, md_ast=get_office_handle(path))
    return None


def _read_text(path: str, format_name: str) -> str:
    try:
        with open(path, encoding='utf-8') as fp:
            return fp.read()
    except UnicodeDecodeError as exc:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_DATA_FORMAT, f'Not a valid {format_name} document: {path}'
        ) from exc


def get_html_handle(path: str) -> bs4.BeautifulSoup:
    doc = bs4.BeautifulSoup(_read_text(path, 'HTML'), 'lxml')
    if doc.find() is None:
        raise excs.RequestError(excs.ErrorCode.INVALID_DATA_FORMAT, f'Not a valid HTML document: {path}')
    return doc


def get_markdown_handle(path: str) -> list:
    Env.get().require_package('mistune', [3, 0])
    import mistune

    text = _read_text(path, 'Markdown')
    md_parser = mistune.create_markdown(renderer=None)
    result = md_parser(text)
    assert isinstance(result, list)
    return result


def get_xml_handle(path: str) -> bs4.BeautifulSoup:
    doc = bs4.BeautifulSoup(_read_text(path, 'XML'), 'xml')
    if doc.find() is None:
        raise excs.RequestError(excs.ErrorCode.INVALID_DATA_FORMAT, f'Not a valid XML document: {path}')
    return doc


def get_pdf_handle(path: str) -> PdfDocument:
    try:
        return PdfDocument(path)
    except PdfiumError as exc:
        raise excs.RequestError(excs.ErrorCode.INVALID_DATA_FORMAT, f'Not a valid PDF document: {path}') from exc


def get_txt(path: str) -> str:
    return _read_text(path, 'Text')


def get_office_handle(path: str) -> list:
    """Convert office documents (PPTX, DOCX, XLSX) to markdown using MarkItDown."""
    Env.get().require_package('mistune', [3, 0])
    Env.get().require_package('markitdown')
    import mistune
    from markitdown import MarkItDown, MarkItDownException, MissingDependencyException

    md = MarkItDown(enable_plugins=False)
    try:
        markdown_text = md.convert(path).text_content
    except MarkItDownException as exc:
        causes = [exc, *(a.exc_info[1] for a in getattr(exc, 'attempts', None) or [] if a.exc_info is not None)]
        missing_dep = next((c for c in causes if isinstance(c, MissingDependencyException)), None)
        if missing_dep is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot read {path}: {missing_dep}') from exc
        raise excs.RequestError(excs.ErrorCode.INVALID_DATA_FORMAT, f'Not a valid Office document: {path}') from exc

    md_parser = mistune.create_markdown(renderer=None)
    result = md_parser(markdown_text)
    assert isinstance(result, list)
    return result
