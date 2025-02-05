# Project params
PROJECT_NAME = 'financial_research_agent'

# Index params
PDF_INDEX_NAME = f'{PROJECT_NAME}.financial_research_reports'
PDF_CHUNKS_NAME = f'{PROJECT_NAME}.financial_research_report_chunks'
EMBEDDING_MODEL = 'intfloat/e5-large-v2'
DOCUMENT_URL = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/'

# Index and agents are persistent. You can delete them with DELETE_ALL=True
DELETE_ALL = True

# Agent params
AGENT_MODEL = 'gpt-4o-mini'
AGENT_NAME = f'{PROJECT_NAME}.openai_gpt_4o_mini'
SYSTEM_PROMPT = """
You are a financial research assistant.

You are given a question and a list of financial research reports.

Your job is to answer the question based on the information provided in the reports.

You should use the search tool to find the most relevant information in the reports.

"""