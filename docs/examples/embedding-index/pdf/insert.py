import pixeltable as pxt

documents_t = pxt.get_table('pdf_index.pdfs')

# Sample PDFs
DOCUMENT_URL = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/'

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

# PDF ingestion pipeline (read, parse, and store)
documents_t.insert({'pdf': url} for url in document_urls)
