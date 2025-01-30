# Website Embedding Index Example

This example demonstrates how to create a semantic search index for website content using Pixeltable. It shows how to:
1. Create a table for website content
2. Split website content into manageable chunks
3. Create embeddings for semantic search
4. Perform similarity searches on the website content

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Steps to Run

1. Run the scripts in order:
   - `create_index.py`: Creates the website table and sets up the embedding index
   - `insert.py`: Inserts website URLs into the table
   - `query.py`: Demonstrates how to perform semantic searches

## What This Example Does

- Creates a table for storing website content
- Splits website content into chunks of approximately 300 tokens each
- Creates embeddings using the E5-large-v2 model from Hugging Face
- Enables semantic search queries on website content

## Code Structure

### create_index.py
- Initializes a new Pixeltable directory
- Creates a table for website documents
- Sets up document chunking using `DocumentSplitter`
- Creates an embedding index using the E5-large-v2 model

### insert.py
- Shows how to insert website URLs into the table
- Automatically triggers content chunking and embedding generation
- Demonstrates with an example website (quotes.toscrape.com)

### query.py
- Demonstrates how to perform semantic searches on the website content
- Shows how to retrieve the most relevant content chunks for a given query
- Includes scoring to show relevance of search results

## Sample Query

You can search through the website content semantically using queries like:
```python
# Get similarity scores and top matching sentences
query_text = "Tell me about the albert einstein report"
sim = sentences_view.text.similarity(query_text)
results = sentences_view.order_by(sim, asc=False).select(sentences_view.text, sim=sim).limit(20).collect()
```

The results will return the most semantically relevant chunks from your website content, making it easy to find specific information across your website collection.

