"""Schema definition for the Context-Aware Discord Bot.

Run once to initialize the database schema:
    python setup_pixeltable.py

Idempotent by default. Set RESET_SCHEMA=true to wipe and recreate.
Imported by bot.py on startup.
"""

import os

import config
import functions

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.string import string_splitter

if os.getenv('RESET_SCHEMA', 'false').lower() == 'true':
    pxt.drop_dir(config.APP_NAMESPACE, force=True)

pxt.create_dir(config.APP_NAMESPACE, if_exists='ignore')

messages = pxt.create_table(
    f'{config.APP_NAMESPACE}.messages',
    {'channel_id': pxt.String, 'username': pxt.String, 'content': pxt.String, 'timestamp': pxt.Timestamp},
    if_exists='ignore',
)

sentences = pxt.create_view(
    f'{config.APP_NAMESPACE}.sentences',
    messages,
    iterator=string_splitter(text=messages.content, separators='sentence'),
    if_exists='ignore',
)

sentences.add_embedding_index(
    'text', idx_name='sentences_text_idx', string_embed=functions.get_embeddings, if_exists='ignore'
)
print('  Messages: table + sentences view + embedding index')

chat = pxt.create_table(
    f'{config.APP_NAMESPACE}.chat',
    {'channel_id': pxt.String, 'question': pxt.String, 'timestamp': pxt.Timestamp},
    if_exists='ignore',
)


@pxt.query
def get_context(question_text: str):
    sim = sentences.text.similarity(string=question_text, idx='sentences_text_idx')
    return (
        sentences.order_by(sim, asc=False)
        .select(text=sentences.text, username=sentences.username, sim=sim)
        .limit(config.SIMILARITY_LIMIT)
    )


chat.add_computed_column(context=get_context(chat.question), if_exists='ignore')
chat.add_computed_column(prompt=functions.create_prompt(chat.context, chat.question), if_exists='ignore')
chat.add_computed_column(
    response=openai.chat_completions(
        messages=[{'role': 'system', 'content': config.SYSTEM_PROMPT}, {'role': 'user', 'content': chat.prompt}],
        model=config.LLM_MODEL,
        model_kwargs={
            'temperature': config.LLM_TEMPERATURE,
            'max_tokens': config.LLM_MAX_TOKENS,
            'presence_penalty': 0.7,
            'frequency_penalty': 0.5,
        },
    )
    .choices[0]
    .message.content,
    if_exists='ignore',
)

print('  Chat: table + context retrieval + prompt + LLM response')
print('\nSchema setup complete.')
