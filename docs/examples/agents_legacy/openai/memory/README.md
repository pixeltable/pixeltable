# Memory-Enabled Agent Example with OpenAI

This example demonstrates how to create a chat agent with memory capabilities using Pixeltable and OpenAI's GPT model. The agent can remember previous conversations and refer back to them in its responses.

## Overview

The example shows how to:
- Create a memory system that stores conversation history
- Process queries with context from previous interactions
- Generate contextual responses using OpenAI's language model
- Maintain persistent memory across chat sessions

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key
```

## Files Structure

- `create_agent.py`: Creates the memory and chat session tables with their computed columns
- `chat.py`: Provides the chat interface for interacting with the memory-enabled agent
- `config.py`: Configuration settings (directory name and OpenAI model)
- `requirements.txt`: Required Python packages

## Usage

1. First, run the agent creation script:
```bash
python create_agent.py
```

2. Then use the chat interface to interact with the agent:
```bash
python chat.py
```

The agent will remember your previous interactions and can refer back to them in its responses. You can ask follow-up questions or refer to previously discussed topics.

## How it Works

The agent processes queries through a memory-aware pipeline:

1. **Memory Storage**:
   - Maintains a table that stores all conversation turns
   - Records both user messages and assistant responses
   - Timestamps each interaction for chronological retrieval

2. **Context Processing**:
   - Retrieves recent conversation history when processing new messages
   - Combines system prompt with conversation context
   - Ensures responses are informed by previous interactions

3. **Chat Interface**:
   - Provides a simple way to interact with the memory-enabled agent
   - Displays responses that can reference previous context

## Configuration

The `config.py` file contains:
- `DIRECTORY`: The Pixeltable directory name for storing the agent
- `OPENAI_MODEL`: The OpenAI model to use for processing queries

## Dependencies

- pixeltable: For creating and managing the agent
- openai: For natural language processing and response generation

## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example:

```bash
export OPENAI_API_KEY=your_api_key
```

The agent uses OpenAI's GPT model to process queries and generate contextually aware responses based on conversation history. 