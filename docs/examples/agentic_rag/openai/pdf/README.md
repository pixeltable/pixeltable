# Pixeltable PDF RAG

An Agentic RAG (Retrieval-Augmented Generation) system that processes PDF files using Pixeltable. The system extracts text from PDFs, creates embeddings for semantic search, and provides an agent interface for querying the PDF content.

## Features

- PDF text extraction and processing
- Semantic search using E5 embeddings 
- Agentic RAG pipeline using OpenAI
- Sentence-level chunking for precise retrieval

## Project Structure

```
├── config.py                  # Configuration settings
├── 01_setup.py                # Initialize directory structure
├── 02_create_pdf_index.py     # PDF processing and indexing
├── 03_create_agent.py         # Agent setup and RAG pipeline
├── chat.py                    # Interactive query interface
├── requirements.txt           # Project dependencies
└── README.md
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Key settings in `config.py`:
- `DIRECTORY`: Base directory for Pixeltable tables
- `DOCUMENT_URL`: Path to sample PDF file
- `EMBEDDING_MODEL`: Model for semantic embeddings
- `OPENAI_MODEL`: Model for text generation

## Usage

The system is set up in sequential steps:

1. Build PDF index and agent table:
```bash
python 01_setup.py
```

2. Insert PDF file and test agent:
```bash
python chat.py
```

The chat script will:
- Load the PDF file
- Process it through the RAG pipeline
- Execute a sample query

## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example. 

```bash
export OPENAI_API_KEY=your_api_key
```

