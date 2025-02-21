import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

DIRECTORY = 'pdf_index'
TABLE_NAME = f'{DIRECTORY}.pdfs'
VIEW_NAME = f'{DIRECTORY}.pdf_chunks'
DELETE_INDEX = False

if DELETE_INDEX:
    pxt.drop_table(TABLE_NAME, force=True)

if TABLE_NAME not in pxt.list_tables():
    # Create documents table
    pxt.create_dir(DIRECTORY, if_exists='ignore')
    pdf_index = pxt.create_table(TABLE_NAME, {'pdf': pxt.Document})

    # Create view that chunks PDFs into sections
    chunks_view = pxt.create_view(
        VIEW_NAME,
        pdf_index,
        iterator=DocumentSplitter.create(document=pdf_index.pdf, separators='token_limit', limit=300),
    )

    # Define the embedding model
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

    # Create embedding index
    chunks_view.add_embedding_index(column='text', string_embed=embed_model)

else:
    pdf_index = pxt.get_table(TABLE_NAME)

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

# Add data to the table
pdf_index.insert({'pdf': url} for url in document_urls)

# Semantic search
query_text = 'Summarize NVIDIA Report'

# Get the chunks view
chunks_view = pxt.get_table(VIEW_NAME)

# Calculate similarity scores between query and chunks
sim = chunks_view.text.similarity(query_text)

# Get top 20 most similar chunks with their scores
results = chunks_view.order_by(sim, asc=False).select(chunks_view.text, sim=sim).limit(5).collect()
print(results['text'])
