# Audio Embedding Index Example

This example demonstrates how to create a semantic search index for audio content using Pixeltable. It shows how to:
1. Create a table for audio files
2. Transcribe audio using Whisper
3. Split transcriptions into sentences
4. Create embeddings for semantic search
5. Perform similarity searches on the audio content

## Prerequisites

1. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or on Windows PowerShell:
   ```powershell
   $env:OPENAI_API_KEY="your_api_key_here"
   ```

## Steps to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the scripts in order:
   - `create_index.py`: Creates the audio table and sets up the embedding index
   - `insert.py`: Inserts a sample audio file from S3
   - `query.py`: Demonstrates how to perform semantic searches

## What This Example Does

- Creates a table for storing audio files
- Uses Whisper to transcribe audio to text
- Splits transcriptions into sentences using spaCy
- Creates embeddings using the E5-large-v2 model
- Enables semantic search queries on audio content

## Sample Query

You can search through the audio content semantically using queries like:
```python
top_k("What is Pixeltable?")
```

