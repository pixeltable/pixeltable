# pip install pixeltable openai sentence-transformers

import pixeltable as pxt
from pixeltable.functions.openai import vision
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import openai

# Initialize Pixeltable with a fresh directory
pxt.drop_dir("agent", force=True)
pxt.create_dir("agent")

############################################################
# Create Image Knowledge Base
############################################################

# Create images table
img_t = pxt.create_table("agent.images", {"image": pxt.Image})

# Image-to-text
img_t.add_computed_column(
    image_description=vision(
        prompt="Describe the image. Be specific on the colors you see.",
        image=img_t.image,
        model="gpt-4o-mini",
    )
)

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
img_t.add_embedding_index(column="image_description", string_embed=embed_model)

############################################################
# Agentic RAG (Query-as-a-Tool)
############################################################


# Retrieval on the image description
@pxt.query
def top_k(query_text: str):
    sim = img_t.image_description.similarity(query_text)
    return (
        img_t.order_by(sim, asc=False)
        .select(img_t.image_description, sim=sim)
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

    # Sample Images
    IMAGE_URL = (
        "https://raw.github.com/pixeltable/pixeltable/release/docs/resources/images/"
    )

    image_urls = [
        IMAGE_URL + doc
        for doc in [
            "000000000030.jpg",
            "000000000034.jpg",
            "000000000042.jpg",
        ]
    ]

    # Image ingestion pipeline
    img_t.insert({"image": url} for url in image_urls)

    chat("What color are the flowers?")
