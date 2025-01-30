import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

# Initialize Pixeltable
pxt.drop_dir('website_index', force=True)
pxt.create_dir('website_index')

# Create website table
websites_t = pxt.create_table("website_index.websites", {"website": pxt.Document})

# Chunking website into sentences
websites_chunks = pxt.create_view(
    "website_index.website_chunks",
    websites_t,
    iterator=DocumentSplitter.create(
        document=websites_t.website, separators="token_limit", limit=300
    ),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Create embedding index
websites_chunks.add_embedding_index(column="text", string_embed=embed_model)
