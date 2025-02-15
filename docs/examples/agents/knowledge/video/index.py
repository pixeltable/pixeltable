import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.video import extract_audio
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

def create_video_index(index_name: str, view_name: str, reset_history: bool = False):
    # Delete index if it exists
    if reset_history:
        pxt.drop_table(index_name, force=True)

    # Create video table with a video and uploaded_at column
    video_index = pxt.create_table(
        index_name, 
        {'video_file': pxt.Video, 'uploaded_at': pxt.Timestamp},
        if_exists='replace'
    )

    # Extract audio from the video
    video_index.add_computed_column(
        audio_extract=extract_audio(video_index.video_file, format='mp3')
    )

    # Transcribe the extracted audio using OpenAI's transcription (whisper-1)
    video_index.add_computed_column(
        transcription=openai.transcriptions(audio=video_index.audio_extract, model='whisper-1')
    )
    video_index.add_computed_column(transcription_text=video_index.transcription.text)

    # Create view that chunks the transcription text into sentences
    view = pxt.create_view(
        view_name,
        video_index,
        iterator=StringSplitter.create(
            text=video_index.transcription_text, 
            separators='sentence'
        ),
        if_exists='replace'
    )

    # Define the embedding model and create the embedding index
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')
    view.add_embedding_index('text', string_embed=embed_model)