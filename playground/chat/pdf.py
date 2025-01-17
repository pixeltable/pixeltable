# pip install pixeltable tiktoken openai sentence-transformers

import pixeltable as pxt
from pixeltable.functions import openai
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

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
documents_chunks.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Create chatbot
############################################################

# Create chatbot table
conversations = pxt.create_table(
    path_str="chatbot.conversations",
    schema_or_df={"prompt": pxt.String},
    if_exists="ignore",
)


# Retrieve the most similar chunks
@pxt.query
def top_k(query_text: str):
    sim = documents_chunks.text.similarity(query_text)
    return (
        documents_chunks.order_by(sim, asc=False)
        .select(documents_chunks.text, sim=sim)
        .limit(5)
    )


conversations.add_computed_column(semantic_search_context=top_k(conversations.prompt))


# Helper function to build the prompt
@pxt.udf
def create_prompt(top_k_list: list[dict], question: str) -> str:
    concat_top_k = "\n\n".join(elt["text"] for elt in reversed(top_k_list))
    return f"""
    QUESTION:

    {question}

    PASSAGES:

    {concat_top_k}
   """


# Build the prompt
conversations.add_computed_column(
    rag_prompt=create_prompt(
        conversations.semantic_search_context, conversations.prompt
    )
)

# Build the OpenAI message
messages = [
    {
        "role": "system",
        "content": "Answer the users question based on the provided passages.",
    },
    {"role": "user", "content": conversations.rag_prompt},
]

# Call OpenAI
conversations.add_computed_column(
    response=openai.chat_completions(model="gpt-4o-mini", messages=messages)
)

# Extract the answer
conversations.add_computed_column(
    answer=conversations.response.choices[0].message.content
)


############################################################
# Lets test it!
############################################################


def chat(message: str) -> str:
    conversations.insert([{"prompt": message}])
    print(conversations.answer.show())


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

    chat("What is the market outlook for Nvidia?")
