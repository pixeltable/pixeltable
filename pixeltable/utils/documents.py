import dataclasses
import os

import bs4
import puremagic
from pypdfium2 import PdfDocument  # type: ignore[import-untyped]

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: bs4.BeautifulSoup | None = None
    md_ast: list | None = None
    pdf_doc: PdfDocument | None = None
    txt_doc: str | None = None


def get_document_handle(path: str) -> DocumentHandle:
    _, extension = os.path.splitext(path)
    handle = get_handle_by_extension(path, extension)
    if handle is not None:
        return handle

    # if no extension, use puremagic to determine the type
    extension = puremagic.from_file(path)
    handle = get_handle_by_extension(path, extension)
    if handle is not None:
        return handle

    raise excs.Error(f'Unrecognized document format: {path}')


def get_handle_by_extension(path: str, extension: str) -> DocumentHandle | None:
    doc_format = ts.DocumentType.DocumentFormat.from_extension(extension)

    try:
        if doc_format == ts.DocumentType.DocumentFormat.HTML:
            return DocumentHandle(doc_format, bs_doc=get_html_handle(path))
        if doc_format == ts.DocumentType.DocumentFormat.MD:
            return DocumentHandle(doc_format, md_ast=get_markdown_handle(path))
        if doc_format == ts.DocumentType.DocumentFormat.PDF:
            return DocumentHandle(doc_format, pdf_doc=PdfDocument(path))
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
    except Exception as exc:
        raise excs.Error(f'An error occurred processing a {doc_format} document: {path}') from exc

    return None


def get_html_handle(path: str) -> bs4.BeautifulSoup:
    with open(path, 'r', encoding='utf8') as fp:
        doc = bs4.BeautifulSoup(fp, 'lxml')
    if doc.find() is None:
        raise excs.Error(f'Not a valid HTML document: {path}')
    return doc


def get_markdown_handle(path: str) -> list:
    Env.get().require_package('mistune', [3, 0])
    import mistune

    with open(path, encoding='utf8') as file:
        text = file.read()
    md_parser = mistune.create_markdown(renderer=None)
    result = md_parser(text)
    assert isinstance(result, list)
    return result


def get_xml_handle(path: str) -> bs4.BeautifulSoup:
    with open(path, 'r', encoding='utf8') as fp:
        doc = bs4.BeautifulSoup(fp, 'xml')
    if doc.find() is None:
        raise excs.Error(f'Not a valid XML document: {path}')
    return doc


def get_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as fp:
        doc = fp.read()
    return doc


def get_office_handle(path: str) -> list:
    """Convert office documents (PPTX, DOCX, XLSX) to markdown using MarkItDown."""
    Env.get().require_package('mistune', [3, 0])
    Env.get().require_package('markitdown')
    import mistune
    from markitdown import MarkItDown

    md = MarkItDown(enable_plugins=False)
    result = md.convert(path)
    markdown_text = result.text_content

    md_parser = mistune.create_markdown(renderer=None)
    result = md_parser(markdown_text)
    assert isinstance(result, list)
    return result
