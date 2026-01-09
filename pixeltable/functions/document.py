"""
Pixeltable UDFs for `DocumentType`.
"""

from typing import Any, Literal

import pixeltable as pxt


def document_splitter(
    document: Any,
    separators: str,
    *,
    elements: list[Literal['text', 'image']] | None = None,
    limit: int | None = None,
    overlap: int | None = None,
    metadata: str = '',
    skip_tags: list[str] | None = None,
    spacy_model: str = 'en_core_web_sm',
    tiktoken_encoding: str | None = 'cl100k_base',
    tiktoken_target_model: str | None = None,
    image_dpi: int = 300,
    image_format: str = 'png',
) -> tuple[type[pxt.iterators.ComponentIterator], dict[str, Any]]:
    """Iterator over chunks of a document. The document is chunked according to the specified `separators`.

    The iterator yields a `text` field containing the text of the chunk, and it may also
    include additional metadata fields if specified in the `metadata` parameter, as explained below.

    Chunked text will be cleaned with `ftfy.fix_text` to fix up common problems with unicode sequences.

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
        spacy_model: Name of the spaCy model to use for sentence segmentation. This parameter is ignored unless
            the `'sentence'` separator is specified.
        image_dpi: DPI to use when extracting images from PDFs. Defaults to 300.
        image_format: format to use when extracting images from PDFs. Defaults to 'png'.

    Examples:
        All these examples assume an existing table `tbl` with a column `doc` of type `pxt.Document`.

        Create a view that splits all documents into chunks of up to 300 tokens:

        >>> pxt.create_view('chunks', tbl, iterator=document_splitter(tbl.doc, separators='token_limit', limit=300))

        Create a view that splits all documents along sentence boundaries, including title and heading metadata:

        >>> pxt.create_view(
        ...     'sentence_chunks',
        ...     tbl,
        ...     iterator=document_splitter(tbl.doc, separators='sentence', metadata='title,heading')
        ... )
    """

    kwargs: dict[str, Any] = {}
    if elements is not None:
        kwargs['elements'] = elements
    if limit is not None:
        kwargs['limit'] = limit
    if overlap is not None:
        kwargs['overlap'] = overlap
    if metadata != '':
        kwargs['metadata'] = metadata
    if skip_tags is not None:
        kwargs['skip_tags'] = skip_tags
    if spacy_model != 'en_core_web_sm':
        kwargs['spacy_model'] = spacy_model
    if tiktoken_encoding != 'cl100k_base':
        kwargs['tiktoken_encoding'] = tiktoken_encoding
    if tiktoken_target_model is not None:
        kwargs['tiktoken_target_model'] = tiktoken_target_model
    if image_dpi != 300:
        kwargs['image_dpi'] = image_dpi
    if image_format != 'png':
        kwargs['image_format'] = image_format
    return pxt.iterators.document.DocumentSplitter._create(document=document, separators=separators, **kwargs)
