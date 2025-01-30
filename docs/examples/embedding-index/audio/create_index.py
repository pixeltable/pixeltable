import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

import spacy

nlp = spacy.load('en_core_web_sm')

# Initialize Pixeltable
pxt.drop_dir('audio_index', force=True)
pxt.create_dir('audio_index')

# Create audio table
audio_index = pxt.create_table('audio_index.audio', {'audio_file': pxt.Audio})

# Create audio-to-text column
audio_index.add_computed_column(transcription=whisper.transcribe(audio=audio_index.audio_file, model='base.en'))

# Create view with sentence chunks
sentences_view = pxt.create_view(
    'audio_index.audio_sentence_chunks',
    audio_index,
    iterator=StringSplitter.create(text=audio_index.transcription.text, separators='sentence'),
)

# Define the embedding model
embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

# Create embedding index
sentences_view.add_embedding_index(column='text', string_embed=embed_model)
