# DuckDuckGo News Agent

This example demonstrates how to create an AI agent that searches for news using DuckDuckGo and processes the results using Anthropic's Claude model.

## Overview

The project consists of several components:
- A news search functionality using DuckDuckGo's API
- Integration with Anthropic's Claude model for processing search results
- A Pixeltable-based storage system for managing prompts and responses

## Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

The following dependencies are needed:
- pixeltable
- anthropic
- duckduckgo-search

## Configuration

The project configuration is stored in `config.py`:
- `DIRECTORY`: The Pixeltable directory where data will be stored
- `ANTHROPIC_MODEL`: The Anthropic model to use for processing (default: 'claude-3-5-sonnet-20240620')

## Setup and Usage

1. First, run the agent creation script:
```bash
python create_agent.py
```

This script will:
- Create a news agent table
- Add the DuckDuckGo search tool
- Set up the processing pipeline with Claude

2. To interact with the agent, use `chat.py`:
```bash
python chat.py
```

The agent will:
1. Take your news-related query
2. Search for relevant news using DuckDuckGo
3. Process the results using Claude
4. Return a summarized response

## Example Usage

```python
import pixeltable as pxt
from config import DIRECTORY

news_agent = pxt.get_table(f"{DIRECTORY}.news")
news_agent.insert(prompt="What's the latest news in Los Angeles?")
```

## Project Structure

- `create_agent.py`: Creates and configures the news agent
- `chat.py`: Interface for interacting with the agent
- `config.py`: Project configuration
- `requirements.txt`: Package dependencies

## Notes

- The DuckDuckGo search is configured to fetch recent news (within the last month)
- Search results are limited to 20 items by default
- The agent processes the search results to provide a coherent summary 
- Make sure you have your Anthropic API key properly configured in your environment before running the example. 

```bash
export ANTHROPIC_API_KEY=your_api_key
```

