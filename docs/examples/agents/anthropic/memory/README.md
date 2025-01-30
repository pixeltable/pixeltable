# Memory-Enabled Agent Example

This example demonstrates how to create a chat agent with memory capabilities using Pixeltable and Anthropic's Claude model. The agent can remember previous conversations and refer back to them in its responses.

## Overview

The example consists of several components:
- A memory table that stores the conversation history
- A chat session table that processes responses with context
- Functions to retrieve recent memory and create contextual messages
- A chat interface for interacting with the agent

## Files

- `config.py` - Configuration settings including directory name and Anthropic model selection
- `create_agent.py` - Creates the memory and chat session tables with their computed columns
- `chat.py` - Provides the chat interface and example usage
- `requirements.txt` - Required dependencies

## How it Works

1. The agent maintains a memory table that stores all conversation turns (both user and assistant messages) with timestamps.
2. When processing a new message:
   - The user's message is stored in memory
   - Recent conversation history is retrieved (last 10 messages)
   - A system prompt and conversation history are combined with the current message
   - The response is generated using Claude and stored back in memory

## Usage

1. First, ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY=your_api_key
```


2. Run the create_agent.py script to set up the tables:
```python
python create_agent.py
```

3. Then you can interact with the agent using the chat interface:
```python
from config import DIRECTORY

import pixeltable as pxt

news_agent = pxt.get_table(f'{DIRECTORY}.news')

news_agent.insert(prompt="What's the latest news in Los Angeles?")
print(news_agent.select(news_agent.tool_output, news_agent.answer).collect())

```

The agent will remember previous interactions and can refer back to them in its responses. For example, you can ask it to recall your name or previous topics of conversation.

## Features

- Persistent memory across chat sessions
- Contextual responses based on conversation history
- Timestamp-based ordering of messages
- Configurable context window (default: last 10 messages)
- Integration with Anthropic's Claude model
- Clean separation of memory storage and chat processing

## Requirements

- pixeltable
- anthropic

