from config import DIRECTORY, EMBEDDING_MODEL

import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

# Create documents table
documents_t = pxt.create_table(f'{DIRECTORY}.pdfs', {'pdf': pxt.Document})

# Chunk PDF into sentences
documents_chunks = pxt.create_view(
    f'{DIRECTORY}.pdf_chunks',
    documents_t,
    iterator=DocumentSplitter.create(document=documents_t.pdf, separators='token_limit', limit=300),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id=EMBEDDING_MODEL)

# Create embedding index
documents_chunks.add_embedding_index(column='text', string_embed=embed_model)
