# Pixeltable Agentic Website RAG Example

This example demonstrates how to build a RAG (Retrieval-Augmented Generation) system for website content using Pixeltable. It crawls web pages, processes their content, and enables semantic search and question-answering capabilities.

We provide a search query to a tool-calling agent so that it can self-retrieve the relevant documents.

## Features
- Website content crawling and extraction
- Semantic search with E5 embeddings (intfloat/e5-large-v2)
- OpenAI-powered conversational interface
- Automatic webpage indexing and retrieval

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your_api_key
```

## Usage

1. Initialize the system and build the website index:
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
│   ├── create_website_index.py # Website crawling pipeline
│   └── create_agent.py         # RAG agent setup
├── chat.py                     # Interactive chat interface
├── requirements.txt            # Project dependencies
└── README.md
```

## Configuration

Key settings in `config.py`:
- `DIRECTORY`: Base directory for Pixeltable tables
- `EMBEDDING_MODEL`: E5 model for semantic embeddings (intfloat/e5-large-v2)
- `OPENAI_MODEL`: OpenAI model for text generation (gpt-4-mini)

## Dependencies

Main requirements:
- `pixeltable`: Core framework for RAG implementation
- `openai`: OpenAI API client
- `sentence-transformers`: Text embeddings
- `tiktoken`: OpenAI tokenizer

## Example Usage

After setup, you can ask questions about the website content through the chat interface:

```python
# Example query
question = "What information is available on the website?"
# The system will process your question and provide an answer based on the crawled content
```

The system will retrieve relevant segments from the indexed web pages and generate a contextual response using OpenAI's language model.
