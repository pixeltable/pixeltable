from config import DIRECTORY, EMBEDDING_MODEL, WHISPER_MODEL

import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

# Create audio table
audio_t = pxt.create_table(f'{DIRECTORY}.audio', {'audio_file': pxt.Audio})

# Audio-to-text
audio_t.add_computed_column(transcription=whisper.transcribe(audio=audio_t.audio_file, model=WHISPER_MODEL))

# Chunk sentences
sentences_view = pxt.create_view(
    f'{DIRECTORY}.audio_sentence_chunks',
    audio_t,
    iterator=StringSplitter.create(text=audio_t.transcription.text, separators='sentence'),
)

# Define embedding model
embed_model = sentence_transformer.using(model_id=EMBEDDING_MODEL)

# Create embedding index
sentences_view.add_embedding_index(column='text', string_embed=embed_model)
