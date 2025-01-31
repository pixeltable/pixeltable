from config import DIRECTORY, EMBEDDING_MODEL, WHISPER_MODEL

import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter


def create_index(index_name: str, persist: bool = True):

    if persist:

        # Create Audio Location table
        audio_files = pxt.create_table(index_name, {'audio_file': pxt.Audio}, if_exists='ignore')

        # Define sentence transformer embedding model
        embed_model = sentence_transformer.using(model_id=EMBEDDING_MODEL)

        # Audio-to-text
        audio_files.add_computed_column(transcription=whisper.transcribe(audio=audio_files.audio_file, model=WHISPER_MODEL), if_exists='ignore')

        # Chunk sentences
        pxt.create_view(
            f'{index_name}_sentence_chunks',
            audio_files,
            iterator=StringSplitter.create(text=audio_files.transcription.text, separators='sentence'),
            if_exists='ignore'
        )

        sentences_view = pxt.get_table(f'{index_name}_sentence_chunks')

        # Create embedding index
        sentences_view.add_embedding_index(column='text', string_embed=embed_model, if_exists='ignore')

    return sentences_view