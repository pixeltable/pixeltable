"""Multimodal chat schema — idempotent by default.

python setup_pixeltable.py
RESET_SCHEMA=true python setup_pixeltable.py
"""

import os

import config
import functions

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.string import string_splitter
from pixeltable.functions.video import extract_audio

if os.getenv('RESET_SCHEMA', 'false').lower() == 'true':
    pxt.drop_dir(config.APP_NAMESPACE, force=True)

pxt.create_dir(config.APP_NAMESPACE, if_exists='ignore')
ns = config.APP_NAMESPACE

docs_table = pxt.create_table(
    f'{ns}.documents',
    {'document': pxt.Document, 'video': pxt.Video, 'audio': pxt.Audio, 'question': pxt.String},
    if_exists='ignore',
)

conversations = pxt.create_table(
    f'{ns}.conversations', {'role': pxt.String, 'content': pxt.String, 'timestamp': pxt.Timestamp}, if_exists='ignore'
)

docs_table.add_computed_column(audio_extract=extract_audio(docs_table.video, format='mp3'), if_exists='ignore')
docs_table.add_computed_column(
    transcription=openai.transcriptions(audio=docs_table.audio_extract, model='whisper-1'), if_exists='ignore'
)
docs_table.add_computed_column(
    audio_transcription=openai.transcriptions(audio=docs_table.audio, model='whisper-1'), if_exists='ignore'
)
docs_table.add_computed_column(audio_transcription_text=docs_table.audio_transcription.text, if_exists='ignore')
docs_table.add_computed_column(transcription_text=docs_table.transcription.text, if_exists='ignore')

chunks_view = pxt.create_view(
    f'{ns}.chunks',
    docs_table,
    iterator=document_splitter(
        document=docs_table.document, separators='sentence', metadata='title,heading,sourceline'
    ),
    if_exists='ignore',
)

transcription_chunks = pxt.create_view(
    f'{ns}.transcription_chunks',
    docs_table,
    iterator=string_splitter(text=docs_table.transcription_text, separators='sentence'),
    if_exists='ignore',
)

audio_chunks = pxt.create_view(
    f'{ns}.audio_chunks',
    docs_table,
    iterator=string_splitter(text=docs_table.audio_transcription_text, separators='sentence'),
    if_exists='ignore',
)

embed = sentence_transformer.using(model_id=config.EMBEDDING_MODEL_ID)
chunks_view.add_embedding_index('text', idx_name='chunks_text_idx', string_embed=embed, if_exists='ignore')
transcription_chunks.add_embedding_index(
    'text', idx_name='transcription_chunks_text_idx', string_embed=embed, if_exists='ignore'
)
audio_chunks.add_embedding_index('text', idx_name='audio_chunks_text_idx', string_embed=embed, if_exists='ignore')


@pxt.query
def get_chat_history():
    return conversations.order_by(conversations.timestamp).select(
        role=conversations.role, content=conversations.content
    )


@pxt.query
def get_relevant_chunks(query_text: str):
    sim = chunks_view.text.similarity(string=query_text, idx='chunks_text_idx')
    return chunks_view.order_by(sim, asc=False).select(chunks_view.text, sim=sim).limit(20)


@pxt.query
def get_relevant_transcript_chunks(query_text: str):
    sim = transcription_chunks.text.similarity(string=query_text, idx='transcription_chunks_text_idx')
    return transcription_chunks.order_by(sim, asc=False).select(transcription_chunks.text, sim=sim).limit(20)


@pxt.query
def get_relevant_audio_chunks(query_text: str):
    sim = audio_chunks.text.similarity(string=query_text, idx='audio_chunks_text_idx')
    return audio_chunks.order_by(sim, asc=False).select(audio_chunks.text, sim=sim).limit(20)


docs_table.add_computed_column(context_doc=get_relevant_chunks(docs_table.question), if_exists='ignore')
docs_table.add_computed_column(context_video=get_relevant_transcript_chunks(docs_table.question), if_exists='ignore')
docs_table.add_computed_column(context_audio=get_relevant_audio_chunks(docs_table.question), if_exists='ignore')
docs_table.add_computed_column(
    prompt=functions.create_prompt(
        docs_table.context_doc, docs_table.context_video, docs_table.context_audio, docs_table.question
    ),
    if_exists='ignore',
)
docs_table.add_computed_column(chat_history=get_chat_history(), if_exists='ignore')
docs_table.add_computed_column(
    messages=functions.create_messages(docs_table.chat_history, docs_table.prompt), if_exists='ignore'
)
docs_table.add_computed_column(
    response=openai.chat_completions(messages=docs_table.messages, model=config.LLM_MODEL), if_exists='ignore'
)
docs_table.add_computed_column(answer=docs_table.response.choices[0].message.content, if_exists='ignore')

if __name__ == '__main__':
    print('Schema setup complete.')
