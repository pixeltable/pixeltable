import dataclasses
import os
from typing import Optional

import bs4
import fitz  # type: ignore[import-untyped]
import puremagic

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: Optional[bs4.BeautifulSoup] = None
    md_ast: Optional[dict] = None
    pdf_doc: Optional[fitz.Document] = None
    txt_doc: Optional[str] = None


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


def get_handle_by_extension(path: str, extension: str) -> Optional[DocumentHandle]:
    doc_format = ts.DocumentType.DocumentFormat.from_extension(extension)

    try:
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
    except Exception as exc:
        raise excs.Error(f'An error occurred processing a {doc_format} document: {path}') from exc

    return None


def get_html_handle(path: str) -> bs4.BeautifulSoup:
    with open(path, 'r', encoding='utf8') as fp:
        doc = bs4.BeautifulSoup(fp, 'lxml')
    if doc.find() is None:
        raise excs.Error(f'Not a valid HTML document: {path}')
    return doc


def get_markdown_handle(path: str) -> dict:
    Env.get().require_package('mistune', [3, 0])
    import mistune

    with open(path, encoding='utf8') as file:
        text = file.read()
    md_ast = mistune.create_markdown(renderer=None)
    return md_ast(text)


def get_pdf_handle(path: str) -> fitz.Document:
    doc = fitz.open(path)
    # check pdf (bc it will work for images)
    if not doc.is_pdf:
        raise excs.Error(f'Not a valid PDF document: {path}')
    # try to read one page
    next(page for page in doc)
    return doc


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
