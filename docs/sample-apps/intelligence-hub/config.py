"""Configuration for the Intelligence Hub."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Pixeltable ────────────────────────────────────────────────────────────────

APP_NAMESPACE = 'intelligence_hub'

# ── Models ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# LLM: uses OpenAI if OPENAI_API_KEY is set, otherwise falls back to llama.cpp (local)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
USE_OPENAI = bool(OPENAI_API_KEY)
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
LLAMA_REPO_ID = os.getenv('LLAMA_REPO_ID', 'bartowski/Llama-3.2-1B-Instruct-GGUF')
LLAMA_REPO_FILENAME = os.getenv('LLAMA_REPO_FILENAME', '*Q4_K_M.gguf')

# ── Scoring ───────────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = float(os.getenv('RELEVANCE_THRESHOLD', '0.7'))
RELEVANCE_KEYWORDS = [
    'AI', 'data', 'pipeline', 'automation', 'multimodal',
    'vector', 'embedding', 'ETL', 'workflow', 'agent',
]

# ── Notification channels (all optional) ─────────────────────────────────────

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ── Google Sheets (optional) ─────────────────────────────────────────────────

GOOGLE_SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS', '')
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID', '')

# ── Seed data (web URLs loaded by default) ───────────────────────────────────

SEED_URLS = [
    {
        'url': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/LICENSE',
        'title': 'Pixeltable License (Apache 2.0)',
    },
]
