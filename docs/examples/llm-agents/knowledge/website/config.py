# Project params
PROJECT_NAME = 'web_research_agent'

# Index params
WEBSITE_INDEX_NAME = f'{PROJECT_NAME}.web_research'
WEBSITE_CHUNKS_NAME = f'{PROJECT_NAME}.web_research_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
WEB_URL = 'https://quotes.toscrape.com/'

# Index and agents are persistent. You can delete them with DELETE_ALL=True
DELETE_ALL = True

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a research assistant.

Answer the user's question based on the information provided in the website.

"""