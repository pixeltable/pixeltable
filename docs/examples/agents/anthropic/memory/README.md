# Memory-Enabled Chat Agent Example

This example demonstrates how to create a chat agent with memory capabilities using Pixeltable. The agent can remember previous conversations and refer back to them in its responses.

## Overview

The example consists of several components:
- A memory table that stores the conversation history
- A chat session table that processes responses with context
- Functions to retrieve recent memory and create contextual messages
- A chat interface for interacting with the agent

## Files

- `config.py` - Configuration settings including directory and model information
- `setup.py` - Script to set up the environment and create necessary tables
- `create_agent.py` - Defines the memory and chat session tables with their computed columns
- `chat.py` - Provides the chat interface and example usage
- `requirements.txt` - Required dependencies

## How it Works

1. The agent maintains a memory table that stores all conversation turns (both user and assistant messages) with timestamps.
2. When processing a new message:
   - The user's message is stored in memory
   - Recent conversation history is retrieved (last 10 messages)
   - A prompt is created combining system instructions, conversation history, and the current message
   - The response is generated and stored back in memory

## Usage

1. First, run the setup script:
```python
python setup.py
```

2. Then you can interact with the agent using the chat interface:
```python
from chat import chat

response = chat("Hi! My name is Alice.")
print(response)
```

The agent will remember previous interactions and can refer back to them in its responses. For example, you can ask it to recall your name or previous topics of conversation.

## Features

- Persistent memory across chat sessions
- Contextual responses based on conversation history
- Timestamp-based ordering of messages
- Configurable context window (default: last 10 messages)
- Clean separation of memory storage and chat processing

## Requirements

- pixeltable
- openai 