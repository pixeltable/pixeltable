# pip install boto3 pixeltable tiktoken openai openai-whisper spacy sentence-transformers
# Install the spacy model before running: python -m spacy download en_core_web_sm

import pixeltable as pxt
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

# Define the embedding model
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
sentences_view.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Lets test it!
############################################################

if __name__ == "__main__":

    # Audio ingestion pipeline
    audio_t.insert(
        [
            {
                "audio_file": "s3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3"
            }
        ]
    )
