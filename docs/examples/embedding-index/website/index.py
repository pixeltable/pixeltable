import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

DIRECTORY = 'website_index'
TABLE_NAME = f'{DIRECTORY}.websites'
VIEW_NAME = f'{DIRECTORY}.website_chunks'
DELETE_INDEX = False

if DELETE_INDEX:
    pxt.drop_table(TABLE_NAME, force=True)

if TABLE_NAME not in pxt.list_tables():
    # Create website table
    pxt.create_dir(DIRECTORY)
    websites_t = pxt.create_table(TABLE_NAME, {'website': pxt.Document})

    # Chunking website into sentences
    websites_chunks = pxt.create_view(
        VIEW_NAME,
        websites_t,
        iterator=DocumentSplitter.create(document=websites_t.website, separators='token_limit', limit=300),
    )

    # Create embeddings
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

    # Create embedding index
    websites_chunks.add_embedding_index(column='text', string_embed=embed_model)

else:
    websites_t = pxt.get_table(TABLE_NAME)
    websites_chunks = pxt.get_table(VIEW_NAME)

# Website ingestion pipeline (read, parse, and store)
websites_t.insert([{'website': 'https://quotes.toscrape.com/'}])

# Define the search query
query_text = 'Tell me about the albert einstein report'

# Calculate similarity scores between query and sentences
sim = websites_chunks.text.similarity(query_text)

# Get top 20 most similar sentences with their scores
results = websites_chunks.order_by(sim, asc=False).select(websites_chunks.text, sim=sim).limit(5).collect()
print(results["text"])
