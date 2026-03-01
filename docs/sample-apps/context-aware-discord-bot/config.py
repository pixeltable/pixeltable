"""Configuration for the Context-Aware Discord Bot."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

APP_NAMESPACE = 'discord_bot'

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

EMBEDDING_MODEL_ID = 'intfloat/e5-large-v2'
LLM_MODEL = 'gpt-4o-mini'

SYSTEM_PROMPT = (
    'You are a helpful personal assistant focused on natural conversation.\n'
    'CORE PRINCIPLES:\n'
    '- Maintain conversational context\n'
    '- Remember user preferences and details\n'
    '- Progress discussions naturally\n'
    '- Be specific and actionable\n'
    '- Stay on topic unless user changes it\n\n'
    'CONVERSATION STYLE:\n'
    '- Friendly and engaging\n'
    '- Clear and concise\n'
    '- Naturally incorporate context\n'
    '- Ask relevant follow-up questions\n'
    '- Provide practical suggestions'
)

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000
SIMILARITY_LIMIT = 10
SIMILARITY_THRESHOLD = 0.3
