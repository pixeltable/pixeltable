from config import DIRECTORY, DOCUMENT_URL

import pixeltable as pxt

from tables.index import create_index
from tables.agent import create_agent

pdf_table_name = f'{DIRECTORY}.pdfs'
agent_table_name = f'{DIRECTORY}.conversations'

# Create if tables do not exist
if pdf_table_name not in pxt.list_tables():
    create_index()

if agent_table_name not in pxt.list_tables():
    create_agent()

# Fetch tables
pdf_table = pxt.get_table(pdf_table_name)
agent_table = pxt.get_table(agent_table_name)

# Insert sample pdfs
document_urls = [DOCUMENT_URL + doc for doc in ['Mclean-Equity-Alphabet.pdf', 'Zacks-Nvidia-Repeport.pdf']]
pdf_table.insert({'pdf': url} for url in document_urls)

# Ask question
question = 'Explain the Nvidia report'
agent_table.insert([{'prompt': question}])

# Show results
print('\nAnswer:', agent_table.answer.show())
