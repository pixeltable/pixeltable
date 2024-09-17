import dataclasses
from typing import Optional

import bs4
import fitz
import puremagic

import pixeltable.type_system as ts
from pixeltable.env import Env


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: Optional[bs4.BeautifulSoup] = None
    md_ast: Optional[dict] = None
    pdf_doc: Optional[fitz.Document] = None

def get_document_handle(path: str) -> Optional[DocumentHandle]:
    doc_format = puremagic.from_file(path)

    if doc_format == '.pdf':
        pdf_doc = get_pdf_handle(path)
        if pdf_doc is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.PDF, pdf_doc=pdf_doc)

    # The other formats are all text-based, so we can open them and read in their contents

    try:
        with open(path, 'r', encoding='utf8') as file:
            contents = file.read()
    except UnicodeDecodeError:
        # Not a valid text file
        return None

    if doc_format == '.html':
        bs_doc = get_html_handle(contents)
        if bs_doc is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.HTML, bs_doc=bs_doc)

    if doc_format == '.md':
        md_ast = get_markdown_handle(contents)
        if md_ast is not None:
            return DocumentHandle(format=ts.DocumentType.DocumentFormat.MD, md_ast=md_ast)

    return None


def get_html_handle(text: str) -> Optional[bs4.BeautifulSoup]:
    try:
        doc = bs4.BeautifulSoup(text, 'html.parser')
        if doc.find() is None:
            return None
        return doc
    except Exception:
        return None

def get_markdown_handle(text: str) -> Optional[dict]:
    Env.get().require_package('mistune')
    import mistune
    try:
        md_ast = mistune.create_markdown(renderer=None)
        return md_ast(text)
    except Exception:
        return None

def get_pdf_handle(path: str) -> Optional[fitz.Document]:
    import fitz  # aka pymupdf
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
