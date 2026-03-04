"""Schema definition for the Intelligence Hub.

Run once to initialize the database schema:
    python setup_pixeltable.py

WARNING: This drops and recreates the namespace on every run.
"""

import config
import functions

import pixeltable as pxt
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.uuid import uuid7

if config.USE_OPENAI:
    from pixeltable.functions.openai import chat_completions
else:
    from pixeltable.functions.llama_cpp import create_chat_completion

import custom_udfs.discord as discord
import custom_udfs.google_sheets as google_sheets
import custom_udfs.slack as slack
import custom_udfs.telegram as telegram

# ── Clean slate ───────────────────────────────────────────────────────────────

pxt.drop_dir(config.APP_NAMESPACE, force=True)
pxt.create_dir(config.APP_NAMESPACE, if_exists='ignore')

# ── 1. Sources Table ─────────────────────────────────────────────────────────

sources = pxt.create_table(
    f'{config.APP_NAMESPACE}.sources',
    {
        'uuid': uuid7(),
        'url': pxt.String,
        'title': pxt.String,
        'doc': pxt.Document,
        'origin': pxt.String,
        'metadata': pxt.Json,
        'timestamp': pxt.Timestamp,
    },
    primary_key=['uuid'],
    if_exists='ignore',
)

print('  Sources: table created')

# ── 2. Document Chunking + Embedding ─────────────────────────────────────────

chunks = pxt.create_view(
    f'{config.APP_NAMESPACE}.chunks',
    sources,
    iterator=document_splitter(sources.doc, separators='sentence,token_limit', limit=300, overlap=50),
    if_exists='ignore',
)

sentence_embed = sentence_transformer.using(model_id=config.EMBEDDING_MODEL_ID)

chunks.add_embedding_index('text', string_embed=sentence_embed, if_exists='ignore')

print('  Chunks: view + embedding index')


@pxt.query
def search_chunks(query_text: str, n: int = 5):
    """Semantic search across all ingested content."""
    sim = chunks.text.similarity(string=query_text)
    return chunks.order_by(sim, asc=False).limit(n).select(chunks.text, chunks.title, sim=sim)


# ── 3. AI Processing (Computed Columns) ──────────────────────────────────────

# Summarization via LLM
if config.USE_OPENAI:
    sources.add_computed_column(
        summary=chat_completions(
            messages=functions.make_summary_prompt(sources.title, sources.origin), model=config.LLM_MODEL
        )
        .choices[0]
        .message.content,
        if_exists='ignore',
    )
    print(f'  LLM: OpenAI ({config.LLM_MODEL})')
else:
    sources.add_computed_column(
        summary=create_chat_completion(
            messages=functions.make_summary_prompt(sources.title, sources.origin),
            repo_id=config.LLAMA_REPO_ID,
            repo_filename=config.LLAMA_REPO_FILENAME,
            model_kwargs={'max_tokens': 256},
        )['choices'][0]['message']['content'],
        if_exists='ignore',
    )
    print(f'  LLM: llama.cpp ({config.LLAMA_REPO_ID})')

# Relevance scoring
sources.add_computed_column(relevance=functions.score_relevance(sources.summary), if_exists='ignore')

# Alert text (shared by all notification channels)
sources.add_computed_column(
    alert_text=functions.format_alert(sources.title, sources.summary, sources.relevance), if_exists='ignore'
)

print('  Processing: summary + relevance + alert_text')

# ── 4. Notification Channels (Custom UDFs as Computed Columns) ───────────────

if config.SLACK_WEBHOOK_URL:
    sources.add_computed_column(
        slack_alert=slack.send_message(config.SLACK_WEBHOOK_URL, sources.alert_text), if_exists='ignore'
    )
    print('  Notifications: Slack enabled')

if config.DISCORD_WEBHOOK_URL:
    sources.add_computed_column(
        discord_alert=discord.send_message(config.DISCORD_WEBHOOK_URL, sources.alert_text), if_exists='ignore'
    )
    print('  Notifications: Discord enabled')

if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
    sources.add_computed_column(
        telegram_alert=telegram.send_message(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, sources.alert_text),
        if_exists='ignore',
    )
    print('  Notifications: Telegram enabled')

# ── 5. Export (Google Sheets) ────────────────────────────────────────────────

if config.GOOGLE_SHEETS_CREDENTIALS and config.GOOGLE_SHEET_ID:
    sources.add_computed_column(
        _export_payload=functions.make_export_row(sources.title, sources.origin, sources.summary, sources.relevance),
        if_exists='ignore',
    )
    export_row = google_sheets.make_export_udf(config.GOOGLE_SHEETS_CREDENTIALS, config.GOOGLE_SHEET_ID, 'Results')
    sources.add_computed_column(sheet_export=export_row(sources._export_payload), if_exists='ignore')
    print('  Export: Google Sheets enabled')

print('\nSchema setup complete.')
