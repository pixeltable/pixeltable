import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

# Initialize Pixeltable
pxt.drop_dir('pdf_index', force=True)
pxt.create_dir('pdf_index')

# Create documents table
documents_t = pxt.create_table("pdf_index.pdfs", {"pdf": pxt.Document})

# Chunk PDF into sentences
documents_chunks = pxt.create_view(
    "pdf_index.pdf_chunks",
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.pdf, separators="token_limit", limit=300
    ),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Create embedding index
documents_chunks.add_embedding_index(column="text", string_embed=embed_model)
