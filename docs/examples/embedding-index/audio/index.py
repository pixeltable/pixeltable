import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

DIRECTORY = 'audio_index'
TABLE_NAME = f'{DIRECTORY}.audio'
VIEW_NAME = f'{DIRECTORY}.audio_sentence_chunks'
DELETE_INDEX = False

if DELETE_INDEX:
    pxt.drop_table(TABLE_NAME, force=True)

if TABLE_NAME not in pxt.list_tables():

    # Create audio table
    pxt.create_dir(DIRECTORY)
    audio_index = pxt.create_table(TABLE_NAME, {'audio_file': pxt.Audio})

    # Create audio-to-text column
    audio_index.add_computed_column(transcription=whisper.transcribe(audio=audio_index.audio_file, model='base.en'))

    # Create view that chunks text into sentences
    sentences_view = pxt.create_view(
        VIEW_NAME,
        audio_index,
        iterator=StringSplitter.create(text=audio_index.transcription.text, separators='sentence'),
    )

    # Define the embedding model
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

    # Create embedding index
    sentences_view.add_embedding_index(column='text', string_embed=embed_model)

else: 
    audio_index = pxt.get_table(TABLE_NAME)


# Add data to the table
audio_index.insert([{'audio_file': 's3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3'}])

# Semantic search
query_text = 'What is Pixeltable?'

# Calculate similarity scores between query and sentences
sim = sentences_view.text.similarity(query_text)

# Get top 20 most similar sentences with their scores
results = sentences_view.order_by(sim, asc=False).select(sentences_view.text, sim=sim).limit(5).collect()
print(results["text"])