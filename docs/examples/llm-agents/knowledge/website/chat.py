import config
import pixeltable as pxt

from pixel.agent import create_agent
from pixel.index import create_index

# Create project
pxt.create_dir(config.PROJECT_NAME, if_exists='ignore')

# Create website index
website_source_table, website_index = create_index(
    index_name=config.WEBSITE_INDEX_NAME,
    chunks_name=config.WEBSITE_CHUNKS_NAME,
    purge=config.DELETE_ALL
)

# Insert sample pdfs
website_urls = [config.WEB_URL]
website_source_table.insert({'website': url} for url in website_urls)

# Create agent
web_research_agent = create_agent(
    agent_name=config.AGENT_NAME,
    index=website_index,
    system_prompt=config.SYSTEM_PROMPT,
    purge=config.DELETE_ALL
)

# Ask question
question = 'Tell me about the albert einstein quote'
web_research_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', web_research_agent.answer.show())
