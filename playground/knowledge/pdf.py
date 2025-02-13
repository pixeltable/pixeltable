# pip install pixeltable tiktoken sentence-transformers

import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer

# Initialize Pixeltable
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

############################################################
# Create PDF Knowledge Base
############################################################

# Create documents table
documents_t = pxt.create_table("chatbot.pdfs", {"pdf": pxt.Document})

# Chunk PDF into sentences
documents_chunks = pxt.create_view(
    "chatbot.pdf_chunks",
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.pdf, separators="token_limit", limit=300
    ),
)

# Define the embedding model
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
documents_chunks.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Lets test it!
############################################################

if __name__ == "__main__":

    # Sample PDFs
    DOCUMENT_URL = (
        "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/"
    )

    document_urls = [
        DOCUMENT_URL + doc
        for doc in [
            "Argus-Market-Digest-June-2024.pdf",
            "Argus-Market-Watch-June-2024.pdf",
            "Company-Research-Alphabet.pdf",
            "Jefferson-Amazon.pdf",
            "Mclean-Equity-Alphabet.pdf",
            "Zacks-Nvidia-Repeport.pdf",
        ]
    ]

    # PDF ingestion pipeline (read, parse, and store)
    documents_t.insert({"pdf": url} for url in document_urls)
