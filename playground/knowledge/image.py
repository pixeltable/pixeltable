# pip install pixeltable openai sentence-transformers

import pixeltable as pxt
from pixeltable.functions.openai import vision
from pixeltable.functions.huggingface import sentence_transformer

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

# Define the embedding model
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
img_t.add_embedding_index(column="image_description", string_embed=embed_model)

############################################################
# Lets test it!
############################################################

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
