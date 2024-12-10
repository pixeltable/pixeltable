import dataclasses
from typing import Optional

import bs4
import fitz  # type: ignore[import-untyped]
import puremagic

import pixeltable.type_system as ts
from pixeltable.env import Env


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: Optional[bs4.BeautifulSoup] = None
    md_ast: Optional[dict] = None
    pdf_doc: Optional[fitz.Document] = None
    txt_doc: Optional[str] = None


def get_document_handle(path: str) -> Optional[DocumentHandle]:
    doc_format = puremagic.from_file(path)

    if doc_format == '.pdf':
        pdf_doc = get_pdf_handle(path)
        if pdf_doc is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.PDF, pdf_doc=pdf_doc)

    if doc_format == '.html':
        bs_doc = get_html_handle(path)
        if bs_doc is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.HTML, bs_doc=bs_doc)

    if doc_format == '.md':
        md_ast = get_markdown_handle(path)
        if md_ast is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.MD, md_ast=md_ast)

    if doc_format == '.xml':
        bs_doc = get_xml_handle(path)
        if bs_doc is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.XML, bs_doc=bs_doc)

    if doc_format == '.txt':
        txt_doc = get_txt(path)
        if txt_doc is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.TXT, txt_doc=txt_doc)

    return None


def get_pdf_handle(path: str) -> Optional[fitz.Document]:
    try:
        doc = fitz.open(path)
        # check pdf (bc it will work for images)
        if not doc.is_pdf:
            return None
        # try to read one page
        next(page for page in doc)
        return doc
    except Exception:
        return None


def get_html_handle(path: str) -> Optional[bs4.BeautifulSoup]:
    try:
        with open(path, 'r', encoding='utf8') as fp:
            doc = bs4.BeautifulSoup(fp, 'lxml')
        return doc if doc.find() is not None else None
    except Exception:
        return None


def get_xml_handle(path: str) -> Optional[bs4.BeautifulSoup]:
    try:
        with open(path, 'r', encoding='utf8') as fp:
            doc = bs4.BeautifulSoup(fp, 'xml')
        return doc if doc.find() is not None else None
    except Exception:
        return None


def get_markdown_handle(path: str) -> Optional[dict]:
    Env.get().require_package('mistune', [3, 0])
    import mistune
    try:
        with open(path, encoding='utf8') as file:
            text = file.read()
        md_ast = mistune.create_markdown(renderer=None)
        return md_ast(text)
    except Exception:
        return None

def get_txt(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            doc = f.read()
        return doc if doc != '' else None
    except Exception:
        return None
