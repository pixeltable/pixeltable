# pip install pixeltable openai sentence-transformers

import pixeltable as pxt
from pixeltable.functions.openai import vision
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import openai

# Initialize Pixeltable with a fresh directory
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

############################################################
# Create Image Knowledge Base
############################################################

# Create images table
img_t = pxt.create_table("chatbot.images", {"image": pxt.Image})

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
# Create chatbot
############################################################

# Create chatbot table
conversations = pxt.create_table(
    path_str="chatbot.conversations",
    schema_or_df={"prompt": pxt.String},
    if_exists="ignore",
)


# Retrieval on the image description
@pxt.query
def top_k(query_text: str):
    sim = img_t.image_description.similarity(query_text)
    return (
        img_t.order_by(sim, asc=False)
        .select(img_t.image_description, sim=sim)
        .limit(10)
    )


conversations.add_computed_column(semantic_search_context=top_k(conversations.prompt))


# Helper function to build the prompt
@pxt.udf
def create_prompt(top_k_list: list[dict], question: str) -> str:
    concat_top_k = "\n\n".join(elt["image_description"] for elt in reversed(top_k_list))
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


# Chatbot
def chat(message: str) -> str:
    conversations.insert([{"prompt": message}])
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
