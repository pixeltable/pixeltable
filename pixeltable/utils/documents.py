from typing import Optional, Dict
import dataclasses

import pixeltable.type_system as ts


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: Optional['bs4.BeautifulSoup'] = None
    md_ast: Optional[Dict] = None
    pdf_doc: Optional['pdfminer.PDFDocument'] = None

def get_document_handle(path: str) -> Optional[DocumentHandle]:
    pdf_doc = get_pdf_handle(path)  # try pdf first, bc correct pdf will fail to be read as text at open()
    if pdf_doc is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.PDF, pdf_doc=pdf_doc)
    bs_doc = get_html_handle(path)
    if bs_doc is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.HTML, bs_doc=bs_doc)
    md_ast = get_markdown_handle(path)
    if md_ast is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.MD, md_ast=md_ast)
    return pdf_doc

def get_html_handle(path: str) -> Optional['bs4.BeautifulSoup']:
    import bs4
    with open(path, 'r') as file:
        s = file.read()
        try:
            doc = bs4.BeautifulSoup(s, 'html.parser')
        except Exception as e:
            return None
        if doc.find() is None:
            return None
        return doc

def get_markdown_handle(path: str) -> Optional[Dict]:
    import mistune
    with open(path, 'r') as file:
        s = file.read()
        try:
            md_ast = mistune.create_markdown(renderer=None)
            return md_ast(s)
        except Exception as e:
            return None

def get_pdf_handle(path : str) -> Optional[str]:
    import pdfminer
    import pdfminer.high_level
    # concepts: https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html#topic-pdf-to-text-layout
    # usage: https://pdfminersix.readthedocs.io/en/latest/tutorial/extract_pages.html

    with open(path, 'rb') as file:
        try:
            page_iter = pdfminer.high_level.extract_pages(file)
            i = -1
            for i, page in enumerate(page_iter):
                break
            if i == -1:
                return None
            return path # for now make a new iterator every time
        except Exception as e:
            return None