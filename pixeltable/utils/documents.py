from typing import Optional, Dict
import dataclasses

import pixeltable.type_system as ts


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: Optional['bs4.BeautifulSoup'] = None
    md_ast: Optional[Dict] = None
    pdf_doc: Optional['fitz.Document'] = None

def get_document_handle(path: str) -> Optional[DocumentHandle]:
    # NB: try pdf first, because correct PDF must be opened in binary mode
    pdf_doc = get_pdf_handle(path)
    if pdf_doc is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.PDF, pdf_doc=pdf_doc)

    # currently the rest of the types are text-based, so we can open them in utf8 mode
    try:
        with open(path, 'r', encoding='utf8') as file:
            contents = file.read()
    except UnicodeDecodeError:
        return None
    md_ast = get_markdown_handle(contents)
    if md_ast is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.MD, md_ast=md_ast)
    # NB: get_html_handle must be last, because the bs4 will return a handle for md files as well,
    # and we want to prefer the md handle.
    bs_doc = get_html_handle(contents)
    if bs_doc is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.HTML, bs_doc=bs_doc)

    return None

def get_html_handle(text: str) -> Optional['bs4.BeautifulSoup']:
    import bs4
    try:
        doc = bs4.BeautifulSoup(text, 'html.parser')
        if doc.find() is None:
            return None
        return doc
    except Exception:
        return None

def get_markdown_handle(text: str) -> Optional[Dict]:
    import mistune
    try:
        md_ast = mistune.create_markdown(renderer=None)
        return md_ast(text)
    except Exception:
        return None

def get_pdf_handle(path : str) -> Optional['fitz.Document']:
    import fitz # aka pymupdf
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