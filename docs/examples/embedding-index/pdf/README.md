# PDF Embedding Index Example

This example demonstrates how to create a semantic search index for PDF documents using Pixeltable. It shows how to:
1. Create a table for PDF files
2. Split PDFs into manageable chunks
3. Create embeddings for semantic search
4. Perform similarity searches on the PDF content

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Steps to Run

1. Run the scripts in order:
   - `create_index.py`: Creates the PDF table and sets up the embedding index
   - `insert.py`: Inserts PDF files into the table
   - `query.py`: Demonstrates how to perform semantic searches

## What This Example Does

- Creates a table for storing PDF documents
- Splits PDFs into chunks of approximately 300 tokens each
- Creates embeddings using the E5-large-v2 model from Hugging Face
- Enables semantic search queries on PDF content

## Code Structure

### create_index.py
- Initializes a new Pixeltable directory
- Creates a table for PDF documents
- Sets up document chunking using `DocumentSplitter`
- Creates an embedding index using the E5-large-v2 model

### insert.py
- Shows how to insert PDF files into the table
- Automatically triggers document chunking and embedding generation

### query.py
- Demonstrates how to perform semantic searches on the PDF content
- Shows how to retrieve the most relevant PDF chunks for a given query

## Sample Query

You can search through the PDF content semantically using queries like:
```python
top_k("What is the main topic of this document?")
```

The results will return the most semantically relevant chunks from your PDF documents, making it easy to find specific information across your document collection.

