# Pixeltable Audio RAG

An Agentic RAG (Retrieval-Augmented Generation) system that processes audio files using Pixeltable. The system transcribes audio, creates embeddings for semantic search, and provides an agent interface for querying the audio content.

## Features

- Audio transcription using Whisper
- Semantic search using E5 embeddings
- Agentic RAG pipeline using OpenAI
- Sentence-level chunking for precise retrieval

## Project Structure

```
├── config.py                   # Configuration settings
├── 01_setup.py                 # Initialize directory structure
├── 02_create_audio_index.py    # Audio processing and indexing
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

3. Install required spacy model:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

Key settings in `config.py`:
- `DIRECTORY`: Base directory for Pixeltable tables
- `AUDIO_FILE`: S3 path to sample audio file
- `EMBEDDING_MODEL`: Model for semantic embeddings
- `WHISPER_MODEL`: Model for audio transcription
- `OPENAI_MODEL`: Model for text generation

## Usage

The system is set up in sequential steps:

1. Build audio index and agent table:
```bash
python 01_setup.py
```
2. Insert audio file and test agent:
```bash
python chat.py
```

The chat script will:
- Load the audio file
- Process it through the RAG pipeline
- Execute a sample query

## Example Query

```python
question = 'Summarize the tour'
agent_table.insert([{'prompt': question}])
```

## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example. 

```bash
export OPENAI_API_KEY=your_api_key
```

