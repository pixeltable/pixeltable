# pip install pixeltable tiktoken openai sentence-transformers

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer

# Initialize Pixeltable
pxt.drop_dir("agent", force=True)
pxt.create_dir("agent")

############################################################
# Create PDF Knowledge Base
############################################################

# Create documents table
documents_t = pxt.create_table("agent.pdfs", {"pdf": pxt.Document})

# Chunk PDF into sentences
documents_chunks = pxt.create_view(
    "agent.pdf_chunks",
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.pdf, separators="token_limit", limit=300
    ),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Create embedding index
documents_chunks.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Agentic RAG (Query-as-a-Tool)
############################################################


# Define Query and Tool
@pxt.query
def top_k(query_text: str):
    sim = documents_chunks.text.similarity(query_text)
    return (
        documents_chunks.order_by(sim, asc=False)
        .select(documents_chunks.text, sim=sim)
        .limit(10)
    )


# Create Tools
tools = pxt.tools(top_k)

############################################################
# Create Agent
############################################################

# Create agent table
conversations = pxt.create_table(
    path_str="agent.conversations",
    schema_or_df={"prompt": pxt.String},
    if_exists="ignore",
)

# Initial tool calling to retrieve documents
messages = [{"role": "user", "content": conversations.prompt}]

conversations.add_computed_column(
    tool_response=openai.chat_completions(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice=tools.choice(required=True),
    )
)

conversations.add_computed_column(
    retrieved_passages=openai.invoke_tools(tools, conversations.tool_response)
)


# Helper function to create final prompt
@pxt.udf
def create_prompt(question: str, passages: list[dict]) -> str:
    return f"""
    QUESTION:

    {question}

    RELEVANT PASSAGES:

    {passages}
    """


# Add final prompt to agent table
conversations.add_computed_column(
    final_prompt=create_prompt(conversations.prompt, conversations.retrieved_passages)
)

# Final response generation
final_messages = [
    {
        "role": "system",
        "content": "Answer the user's question based on the retrieved passages.",
    },
    {"role": "user", "content": conversations.final_prompt},
]

conversations.add_computed_column(
    final_response=openai.chat_completions(model="gpt-4o-mini", messages=final_messages)
)

conversations.add_computed_column(
    answer=conversations.final_response.choices[0].message.content
)

############################################################
# Chat Interface
############################################################


def chat(message: str) -> str:
    conversations.insert([{"prompt": message}])
    print(conversations.retrieved_passages.show())
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

    # PDF ingestion pipeline
    documents_t.insert({"pdf": url} for url in document_urls)

    chat("What is the market outlook for Nvidia?")
