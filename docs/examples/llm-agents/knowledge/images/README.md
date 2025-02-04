# Pixeltable Agentic PDF RAG Example

This example demonstrates how to build a RAG (Retrieval-Augmented Generation) system for PDF documents using Pixeltable. It processes PDF files, extracts their content, and enables semantic search and question-answering capabilities.

We provide a search query to a tool-calling agent so that it can self-retrieve the relevant documents.

## Features
- PDF content extraction and processing
- Semantic search with E5 embeddings (intfloat/e5-large-v2)
- OpenAI-powered conversational interface
- Automatic document indexing and retrieval

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set OpenAI API key
export OPENAI_API_KEY=your_api_key
```

## Usage

1. Initialize the system and build the document index:
```bash
python setup.py
```

2. Start the interactive chat interface:
```bash
python chat.py
```

## Project Structure

```
├── config.py                   # Configuration settings
├── setup.py                    # System initialization
├── tables/
│   ├── create_pdf_index.py     # PDF processing pipeline
│   └── create_agent.py         # RAG agent setup
├── chat.py                     # Interactive chat interface
├── requirements.txt            # Project dependencies
└── README.md
```

## Configuration

Key settings in `config.py`:
- `DIRECTORY`: Base directory for Pixeltable tables
- `DOCUMENT_URL`: URL to the sample PDF documents
- `EMBEDDING_MODEL`: E5 model for semantic embeddings (intfloat/e5-large-v2)
- `OPENAI_MODEL`: OpenAI model for text generation (gpt-4-mini)

## Dependencies

Main requirements:
- `pixeltable`: Core framework for RAG implementation
- `openai`: OpenAI API client
- `sentence-transformers`: Text embeddings
- `tiktoken`: OpenAI tokenizer

## Example Usage

After setup, you can ask questions about the PDF content through the chat interface:

```python
# Example query
question = "What are the key concepts discussed in the document?"
# The system will process your question and provide an answer based on the PDF content
```

The system will retrieve relevant segments from the PDF documents and generate a contextual response using OpenAI's language model. 
