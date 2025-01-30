import pixeltable as pxt

# Get the sentences view containing the audio chunks
sentences_view = pxt.get_table('website_index.website_chunks')

# Define the search query
query_text = 'Tell me about the albert einstein report'

# Calculate similarity scores between query and sentences
sim = sentences_view.text.similarity(query_text)

# Get top 20 most similar sentences with their scores
results = sentences_view.order_by(sim, asc=False).select(sentences_view.text, sim=sim).limit(20).collect()

print('\nTop matching sentences:')
print('----------------------')
for row in results:
    print(f'\nScore: {row["sim"]:.3f}')
    print(f'Text:  {row["text"]}')
