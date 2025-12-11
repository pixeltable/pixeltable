from typing import Any

from typing_extensions import Literal

import pixeltable as pxt


def document_splitter(
    document: Any,
    separators: str,
    *,
    elements: list[Literal['text', 'image']] | None = None,
    limit: int | None = None,
    overlap: int | None = None,
    metadata: str = '',
    html_skip_tags: list[str] | None = None,
    tiktoken_encoding: str | None = 'cl100k_base',
    tiktoken_target_model: str | None = None,
    image_dpi: int = 300,
    image_format: str = 'png',
) -> tuple[type[pxt.iterators.ComponentIterator], dict[str, Any]]:
    """Iterator over chunks of a document. The document is chunked according to the specified `separators`.

    The iterator yields a `text` field containing the text of the chunk, and it may also
    include additional metadata fields if specified in the `metadata` parameter, as explained below.

    Chunked text will be cleaned with `ftfy.fix_text` to fix up common problems with unicode sequences.

    How to init the `DocumentSplitter` class?

    Args:
        separators: separators to use to chunk the document. Options are:
             `'heading'`, `'paragraph'`, `'sentence'`, `'token_limit'`, `'char_limit'`, `'page'`.
             This may be a comma-separated string, e.g., `'heading,token_limit'`.
        elements: list of elements to extract from the document. Options are:
            `'text'`, `'image'`. Defaults to `['text']` if not specified. The `'image'` element is only supported
            for the `'page'` separator on PDF documents.
        limit: the maximum number of tokens or characters in each chunk, if `'token_limit'`
             or `'char_limit'` is specified.
        metadata: additional metadata fields to include in the output. Options are:
             `'title'`, `'heading'` (HTML and Markdown), `'sourceline'` (HTML), `'page'` (PDF), `'bounding_box'`
             (PDF). The input may be a comma-separated string, e.g., `'title,heading,sourceline'`.
        image_dpi: DPI to use when extracting images from PDFs. Defaults to 300.
        image_format: format to use when extracting images from PDFs. Defaults to 'png'.
    """

    return pxt.iterators.document.DocumentSplitter._create(
        document=document,
        separators=separators,
        elements=elements,
        limit=limit,
        overlap=overlap,
        metadata=metadata,
        html_skip_tags=html_skip_tags,
        tiktoken_encoding=tiktoken_encoding,
        tiktoken_target_model=tiktoken_target_model,
        image_dpi=image_dpi,
        image_format=image_format,
    )
