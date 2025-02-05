import config
import pixeltable as pxt

from pixel.agent import create_agent
from pixel.index import create_index

# Create project
pxt.create_dir(config.PROJECT_NAME, if_exists='ignore')

# Create pdf index
pdf_source_table, pdf_index = create_index(
    index_name=config.PDF_INDEX_NAME,
    chunks_name=config.PDF_CHUNKS_NAME,
    purge=config.DELETE_ALL
)

# Insert sample pdfs
document_urls = [config.DOCUMENT_URL + doc for doc in ['Mclean-Equity-Alphabet.pdf', 'Zacks-Nvidia-Repeport.pdf']]
pdf_source_table.insert({'pdf': url} for url in document_urls)

# Create agent
financial_research_agent = create_agent(
    agent_name=config.AGENT_NAME,
    index=pdf_index,
    system_prompt=config.SYSTEM_PROMPT,
    purge=config.DELETE_ALL
)

# Ask question
question = 'Explain the Nvidia report'
financial_research_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', financial_research_agent.answer.show())
