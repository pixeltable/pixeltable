# Pixeltable Website RAG

An Agentic RAG (Retrieval-Augmented Generation) system that processes website content using Pixeltable. The system extracts content from websites, creates embeddings for semantic search, and provides an agent interface for querying the website content.

## Features

- Website content extraction and processing
- Semantic search using E5 embeddings
- Agentic RAG pipeline using OpenAI
- Token-based chunking for precise retrieval
- Interactive query interface

## Project Structure

```
├── config.py                   # Configuration settings
├── 01_setup.py                 # Initialize directory structure
├── 02_create_website_index.py  # Website processing and indexing
├── 03_create_agent.py          # Agent setup and RAG pipeline
├── chat.py                     # Interactive query interface
├── requirements.txt            # Project dependencies
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
- `EMBEDDING_MODEL`: E5 model for semantic embeddings (intfloat/e5-large-v2)
- `GPT_MODEL`: OpenAI model for text generation

## Usage

The system is set up in sequential steps:

1. Initialize the environment and run the pipeline:
```bash
python 01_setup.py
```

This script will:
- Set up the directory structure
- Create the website index
- Configure the agent

2. Query the website content:
```bash
python chat.py
```

The chat script demonstrates:
- Inserting a sample website (quotes.toscrape.com)
- Processing it through the RAG pipeline
- Executing a sample query

## How It Works

1. **Website Processing (`02_create_website_index.py`)**:
   - Creates a table for storing website content
   - Chunks the content using token-based splitting
   - Generates embeddings using the E5 model
   - Creates an embedding index for semantic search

2. **Agent Setup (`03_create_agent.py`)**:
   - Configures an agent with website search capabilities
   - Sets up a RAG pipeline with:
     - Query generation
     - Semantic search
     - Response synthesis

3. **Interactive Querying (`chat.py`)**:
   - Provides an interface to insert websites
   - Allows querying the processed content
   - Returns AI-generated responses based on retrieved content

## Example Query

```python
question = 'Explain the Albert Einstein quote'
agent_table.insert([{'prompt': question}])
```

The system will:
1. Process the question
2. Search for relevant content
3. Generate a comprehensive response based on the retrieved information


## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example. 

```bash
export OPENAI_API_KEY=your_api_key
```

