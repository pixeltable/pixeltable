# Simple Chatbot with Persistent Storage

A basic chatbot example using Pixeltable and Anthropic that demonstrates how to create a simple chat interface with persistent storage.

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
python 01_setup.py
```

2. Run the chatbot:
```bash
python chat.py
```

## Note

Make sure you have your Anthropic API key properly configured in your environment before running the example. 

```bash
export ANTHROPIC_API_KEY=your_api_key
```

