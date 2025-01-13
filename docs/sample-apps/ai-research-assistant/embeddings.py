from typing import Optional
import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
from data_loader import setup_document_table

# Constants
EMBEDDING_MODEL = 'all-mpnet-base-v2'

def setup_embedding_index():
    """Create embedding index for document chunks."""
    docs_table = setup_document_table()

    chunks_view = pxt.create_view(
        'research.chunks',
        docs_table,
        iterator=DocumentSplitter.create(
            document=docs_table.document,
            separators='sentence',
            metadata='title,heading,sourceline'
        )
    )

    chunks_view.add_embedding_index(
        'text',
        string_embed=sentence_transformer.using(model_id=EMBEDDING_MODEL)
    )

    return chunks_view