import pixeltable as pxt

from index import create_index
from agent import create_agent

# Project params
PROJECT_NAME = 'financial_research_agent'

# Index params
PDF_INDEX_NAME = f'{PROJECT_NAME}.financial_research_reports'
PDF_CHUNKS_NAME = f'{PROJECT_NAME}.financial_research_report_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
DOCUMENT_URL = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/'

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a financial research assistant.

You are given a question and a list of financial research reports.

Your job is to answer the question based on the information provided in the reports.

You should use the search tool to find the most relevant information in the reports.

"""

# Create project
pxt.create_dir(PROJECT_NAME, if_exists='ignore')

# Create pdf index
create_index(
    index_name=PDF_INDEX_NAME,
    chunks_name=PDF_CHUNKS_NAME,
    reset_history=False
)

# The base table holds metadata about the pdfs
pdf_source_table = pxt.get_table(PDF_INDEX_NAME)

# Insert sample pdfs
document_urls = [DOCUMENT_URL + doc for doc in ['Mclean-Equity-Alphabet.pdf', 'Zacks-Nvidia-Repeport.pdf']]
pdf_source_table.insert({'pdf': url} for url in document_urls)

# The index holds the embeddings of the pdfs and the chunked text to retrieve
pdf_index = pxt.get_table(PDF_INDEX_NAME)

# Create agent
create_agent(
    agent_name=AGENT_NAME,
    index=pdf_index,
    llm_model_name=AGENT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    reset_history=False
)

# Ask question
financial_research_agent = pxt.get_table(AGENT_NAME)
question = 'Explain the Nvidia report'
financial_research_agent.insert([{'prompt': question}])

# Show results
print('\nAnswer:', financial_research_agent.answer.show())
