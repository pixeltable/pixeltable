"""Schema definition for the Context-Aware Discord Bot.

Run once to initialize the database schema:
    python setup_pixeltable.py

Can also be imported and called from bot.py on startup.
"""

import numpy as np

import config
import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter


# ── Clean slate ───────────────────────────────────────────────────────────────

pxt.drop_dir(config.APP_NAMESPACE, force=True)
pxt.create_dir(config.APP_NAMESPACE, if_exists='ignore')


# ── 1. Messages Table + Sentence View + Embedding Index ──────────────────────

messages = pxt.create_table(
    f'{config.APP_NAMESPACE}.messages',
    {
        'channel_id': pxt.String,
        'username': pxt.String,
        'content': pxt.String,
        'timestamp': pxt.Timestamp,
    },
    if_exists='ignore',
)

sentences = pxt.create_view(
    f'{config.APP_NAMESPACE}.sentences',
    messages,
    iterator=StringSplitter.create(text=messages.content, separators='sentence'),
    if_exists='ignore',
)


@pxt.expr_udf
def get_embeddings(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id=config.EMBEDDING_MODEL_ID)


sentences.add_embedding_index('text', string_embed=get_embeddings, if_exists='ignore')

print('  Messages: table + sentences view + embedding index')


# ── 2. Chat Table + Computed Columns ─────────────────────────────────────────

chat = pxt.create_table(
    f'{config.APP_NAMESPACE}.chat',
    {
        'channel_id': pxt.String,
        'question': pxt.String,
        'timestamp': pxt.Timestamp,
    },
    if_exists='ignore',
)


@pxt.query
def get_context(question_text: str):
    sim = sentences.text.similarity(question_text)
    return (
        sentences
        .order_by(sim, asc=False)
        .select(text=sentences.text, username=sentences.username, sim=sim)
        .limit(config.SIMILARITY_LIMIT)
    )


chat.add_computed_column(context=get_context(chat.question), if_exists='ignore')


@pxt.udf
def create_prompt(context: list[dict], question: str) -> str:
    context_str = '\n'.join(
        f'{msg["username"]}: {msg["text"]}'
        for msg in context
        if msg['sim'] > config.SIMILARITY_THRESHOLD
    )
    return f'Context:\n{context_str}\n\nQuestion: {question}'


chat.add_computed_column(prompt=create_prompt(chat.context, chat.question), if_exists='ignore')

chat.add_computed_column(
    response=openai.chat_completions(
        messages=[
            {'role': 'system', 'content': config.SYSTEM_PROMPT},
            {'role': 'user', 'content': chat.prompt},
        ],
        model=config.LLM_MODEL,
        model_kwargs={
            'temperature': config.LLM_TEMPERATURE,
            'max_tokens': config.LLM_MAX_TOKENS,
            'presence_penalty': 0.7,
            'frequency_penalty': 0.5,
        },
    ).choices[0].message.content,
    if_exists='ignore',
)

print('  Chat: table + context retrieval + prompt + LLM response')
print('\nSchema setup complete.')
