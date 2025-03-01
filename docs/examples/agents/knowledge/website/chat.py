import pixeltable as pxt

from agent import create_agent
from index import create_index

# Project params
PROJECT_NAME = 'web_research_agent'

# Index params
WEBSITE_INDEX_NAME = f'{PROJECT_NAME}.web_research'
WEBSITE_CHUNKS_NAME = f'{PROJECT_NAME}.web_research_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'

# Index and agents are persistent. You can delete them with DELETE_ALL=True
DELETE_ALL = True

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a research assistant.

Answer the user's question based on the information provided in the website.
"""

# Create project
pxt.create_dir(PROJECT_NAME, if_exists='ignore')

# Create website index
website_source_table, website_index = create_index(
    index_name=WEBSITE_INDEX_NAME,
    chunks_name=WEBSITE_CHUNKS_NAME,
    reset_history=DELETE_ALL
)

# Insert sample pdfs
website_urls = ['https://quotes.toscrape.com/']
website_source_table.insert({'website': url} for url in website_urls)

# Create agent
create_agent(
    agent_name=AGENT_NAME,
    index=website_index,
    system_prompt=SYSTEM_PROMPT,
    reset_history=DELETE_ALL
)

# Ask question
web_research_agent = pxt.get_table(AGENT_NAME)
question = 'Tell me about the albert einstein quote'
web_research_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', web_research_agent.answer.show())
