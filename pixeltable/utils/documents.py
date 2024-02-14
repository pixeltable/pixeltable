from typing import Optional, Dict
import dataclasses

import pixeltable.type_system as ts


@dataclasses.dataclass
class DocumentHandle:
    format: ts.DocumentType.DocumentFormat
    bs_doc: Optional['bs4.BeautifulSoup'] = None
    md_ast: Optional[Dict] = None


def get_document_handle(s: str) -> Optional[DocumentHandle]:
    bs_doc = get_html_handle(s)
    if bs_doc is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.HTML, bs_doc=bs_doc)
    md_ast = get_markdown_handle(s)
    if md_ast is not None:
        return DocumentHandle(format=ts.DocumentType.DocumentFormat.MD, md_ast=md_ast)
    return None

def get_html_handle(s: str) -> Optional['bs4.BeautifulSoup']:
    import bs4
    try:
        doc = bs4.BeautifulSoup(s, 'html.parser')
    except Exception as e:
        return None
    if doc.find() is None:
        return None
    return doc

def get_markdown_handle(s: str) -> Optional[Dict]:
    import mistune
    try:
        md_ast = mistune.create_markdown(renderer=None)
        return md_ast(s)
    except Exception as e:
        return None
