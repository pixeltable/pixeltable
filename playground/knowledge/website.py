# pip install pixeltable tiktoken sentence-transformers

import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer

# Initialize Pixeltable
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

############################################################
# Create Website Knowledge Base
############################################################

# Create website table
websites_t = pxt.create_table("chatbot.websites", {"website": pxt.Document})

# Chunking website into sentences
websites_chunks = pxt.create_view(
    "chatbot.website_chunks",
    websites_t,
    iterator=DocumentSplitter.create(
        document=websites_t.website, separators="token_limit", limit=300
    ),
)

# Define the embedding model
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
websites_chunks.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Lets test it!
############################################################
if __name__ == "__main__":

    # Website ingestion pipeline
    websites_t.insert([{"website": "https://quotes.toscrape.com/"}])
