# Pixeltable Agentic Audio RAG Example

This example demonstrates how to build a RAG (Retrieval-Augmented Generation) system for audio content using Pixeltable. It processes audio files using OpenAI's Whisper for transcription and enables semantic search and question-answering capabilities. 

We provide a search query to a tool-calling agent so that it can self-retreive the relevant documents.

## Features
- Audio transcription using OpenAI's Whisper
- Semantic search with E5 embeddings (intfloat/e5-large-v2)
- OpenAI-powered conversational interface
- Automatic audio content indexing and retrieval

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

1. Build the agent and audio index:
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
│   ├── create_audio_index.py   # Audio processing pipeline
│   └── create_agent.py         # RAG agent setup
├── chat.py                     # Interactive chat interface
├── requirements.txt            # Project dependencies
└── README.md
```

## Configuration

Key settings in `config.py`:
- `DIRECTORY`: Base directory for Pixeltable tables
- `AUDIO_FILE`: S3 path to the sample audio file (Pixeltable tour)
- `EMBEDDING_MODEL`: E5 model for semantic embeddings (intfloat/e5-large-v2)
- `WHISPER_MODEL`: Model for audio transcription (base.en)
- `OPENAI_MODEL`: OpenAI model for text generation (gpt-4-mini)

## Dependencies

Main requirements:
- `pixeltable`: Core framework for RAG implementation
- `openai`: OpenAI API client
- `openai-whisper`: Audio transcription
- `sentence-transformers`: Text embeddings
- `spacy`: Text processing
- `boto3`: S3 file access

## Example Usage

After setup, you can ask questions about the audio content through the chat interface:

```python
# Example query
question = "What are the main features of Pixeltable?"
# The system will process your question and provide an answer based on the audio content
```

The system will retrieve relevant segments from the audio transcription and generate a contextual response using OpenAI's language model.
