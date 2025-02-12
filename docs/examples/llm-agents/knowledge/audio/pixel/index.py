import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

def create_index(index_name: str, chunks_name: str, purge: bool = False) -> tuple[pxt.Table, pxt.Table]:
    # Delete index if it exists
    if purge:
        pxt.drop_table(index_name, force=True)

    # Create index if it doesn't exist
    if index_name not in pxt.list_tables():

        audio_table = pxt.create_table(index_name, {'audio_file': pxt.Audio})

        audio_table.add_computed_column(transcription=whisper.transcribe(audio=audio_table.audio_file, model='base.en'))

        # Create view that chunks PDFs into sections
        chunks_view = pxt.create_view(
            chunks_name,
            audio_table,
            iterator=StringSplitter.create(text=audio_table.transcription.text, separators='sentence'),
        )

        # Define the embedding model
        embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

        # Create embedding index
        chunks_view.add_embedding_index(column='text', string_embed=embed_model)

        return audio_table, chunks_view
    
    else:
        audio_table = pxt.get_table(index_name)
        chunks_view = pxt.get_table(chunks_name)
        return audio_table, chunks_view
