from config import DIRECTORY, EMBEDDING_MODEL

import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

# Create website table
websites_t = pxt.create_table(f'{DIRECTORY}.websites', {'website': pxt.Document})

# Chunking website into sentences
websites_chunks = pxt.create_view(
    f'{DIRECTORY}.website_chunks',
    websites_t,
    iterator=DocumentSplitter.create(document=websites_t.website, separators='token_limit', limit=300),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id=EMBEDDING_MODEL)

# Create embedding index
websites_chunks.add_embedding_index(column='text', string_embed=embed_model)
