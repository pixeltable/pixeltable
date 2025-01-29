# DuckDuckGo News Agent

This example demonstrates how to create an AI agent that searches for news using DuckDuckGo and processes the results using OpenAI's GPT model.

## Overview

The project consists of several components:
- A news search functionality using DuckDuckGo's API
- Integration with OpenAI's GPT model for processing search results
- A Pixeltable-based storage system for managing prompts and responses

## Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

The following dependencies are needed:
- pixeltable
- openai
- duckduckgo-search

## Configuration

The project configuration is stored in `config.py`:
- `DIRECTORY`: The Pixeltable directory where data will be stored
- `OPENAI_MODEL`: The OpenAI model to use for processing (default: 'gpt-4o-mini')

## Setup and Usage

1. First, run the setup script to initialize the environment:
```bash
python 01_setup.py
```

2. The setup script will automatically execute `02_create_agent.py`, which:
   - Creates a news agent table
   - Adds the DuckDuckGo search tool
   - Sets up the processing pipeline with OpenAI

3. To interact with the agent, use `chat.py`:
```bash
python chat.py
```

The agent will:
1. Take your news-related query
2. Search for relevant news using DuckDuckGo
3. Process the results using OpenAI
4. Return a summarized response

## Example Usage

```python
import pixeltable as pxt
from config import DIRECTORY

news_agent = pxt.get_table(f"{DIRECTORY}.news")
news_agent.insert(prompt="What's the latest news in Los Angeles?")
```

## Project Structure

- `01_setup.py`: Initializes the Pixeltable environment
- `02_create_agent.py`: Creates and configures the news agent
- `chat.py`: Interface for interacting with the agent
- `config.py`: Project configuration
- `requirements.txt`: Package dependencies

## Notes

- The DuckDuckGo search is configured to fetch recent news (within the last month)
- Search results are limited to 20 items by default
- The agent processes the search results to provide a coherent summary 
- Make sure you have your OpenAI API key properly configured in your environment before running the example. 

```bash
export OPENAI_API_KEY=your_api_key
```

