from config import DIRECTORY, DOCUMENT_URL

import pixeltable as pxt

# Get tables
pdf_table = pxt.get_table(f'{DIRECTORY}.pdfs')
agent_table = pxt.get_table(f'{DIRECTORY}.conversations')

# Insert sample pdfs
document_urls = [
    DOCUMENT_URL + doc
    for doc in [
        'Argus-Market-Digest-June-2024.pdf',
        'Argus-Market-Watch-June-2024.pdf',
        'Company-Research-Alphabet.pdf',
        'Jefferson-Amazon.pdf',
        'Mclean-Equity-Alphabet.pdf',
        'Zacks-Nvidia-Repeport.pdf',
    ]
]

# PDF ingestion pipeline
pdf_table.insert({'pdf': url} for url in document_urls)

# Ask question
question = 'Explain the Nvidia report'
agent_table.insert([{'prompt': question}])

# Show results
print('\nAnswer:', agent_table.answer.show())
