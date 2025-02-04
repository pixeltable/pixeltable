# Simple Chatbot with Persistent Storage

A basic chatbot example using Pixeltable and OpenAI that demonstrates how to create a simple chat interface with persistent storage.

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the setup script to create the database and tables:
```bash
python create_chat_table.py
```

2. Run the chatbot:
```bash
python chat.py
```

## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example. 

```bash
export OPENAI_API_KEY=your_api_key
```

