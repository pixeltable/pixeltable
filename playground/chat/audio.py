# pip install boto3 pixeltable tiktoken openai openai-whisper spacy sentence-transformers
# Install the spacy model before running: python -m spacy download en_core_web_sm

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

import spacy

nlp = spacy.load("en_core_web_sm")

# Initialize Pixeltable
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

############################################################
# Create Audio Knowledge Base
############################################################

# Create audio table
audio_t = pxt.create_table("chatbot.audio", {"audio_file": pxt.Audio})

# Audio-to-text
audio_t.add_computed_column(
    transcription=whisper.transcribe(audio=audio_t.audio_file, model="base.en")
)

# Split sentences
sentences_view = pxt.create_view(
    "chatbot.audio_sentence_chunks",
    audio_t,
    iterator=StringSplitter.create(
        text=audio_t.transcription.text, separators="sentence"
    ),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
sentences_view.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Create chatbot
############################################################

# Create chatbot table
conversations = pxt.create_table(
    path_str="chatbot.conversations",
    schema_or_df={"prompt": pxt.String},
    if_exists="ignore",
)


# Retrieve the most similar audio chunks
@pxt.query
def top_k(query_text: str):
    sim = sentences_view.text.similarity(query_text)
    return (
        sentences_view.order_by(sim, asc=False)
        .select(sentences_view.text, sim=sim)
        .limit(10)
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


# Chatbot
def chat(message: str) -> str:
    conversations.insert([{"prompt": message}])
    return conversations.answer.show()


if __name__ == "__main__":

    # Audio ingestion pipeline
    audio_t.insert(
        [
            {
                "audio_file": "s3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3"
            }
        ]
    )

    print(chat("Summarize the tour"))
