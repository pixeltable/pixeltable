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
- `DIRECTORY`: The Pixeltable directory where data will be stored (default: 'agent')
- `OPENAI_MODEL`: The OpenAI model to use for processing (default: 'gpt-4o-mini')

## Setup and Usage

1. Create a new environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run `create_agent.py` to set up the news agent:
```bash
python create_agent.py
```

This script will:
- Initialize a fresh Pixeltable environment
- Create a news agent table with the necessary columns
- Add the DuckDuckGo search tool
- Configure the OpenAI processing pipeline

3. To interact with the agent, use `chat.py`:
```bash
python chat.py
```

The agent follows this workflow:
1. Takes your news-related query
2. Searches for relevant news using DuckDuckGo
3. Processes the results using OpenAI
4. Returns both the raw search results and a summarized response

## Example Usage

```python
import pixeltable as pxt
from config import DIRECTORY

# Get the news agent table
news_agent = pxt.get_table(f'{DIRECTORY}.news')

# Make a query
news_agent.insert(prompt="What's the latest news in Los Angeles?")

# Get results and answer
results = news_agent.select(news_agent.tool_output, news_agent.answer).collect()
print(results)
```

## Project Structure

- `create_agent.py`: Creates and configures the news agent with DuckDuckGo search and OpenAI processing
- `chat.py`: Simple interface for interacting with the agent
- `config.py`: Project configuration settings
- `requirements.txt`: Package dependencies

## Features

- DuckDuckGo news search configured with:
  - Global region ('wt-wt')
  - Safe search disabled
  - Time limit set to last month
  - Configurable maximum results (default: 20)
- Structured search results including:
  - Article title
  - Source
  - Publication date
  - Content snippet
- Two-stage OpenAI processing:
  - Initial processing of the user's query
  - Final summarization of search results

## Notes

- Make sure you have your OpenAI API key properly configured in your environment before running the example:

```bash
export OPENAI_API_KEY=your_api_key
```

- The agent stores all interactions in a Pixeltable table, allowing for easy access to historical queries and responses
- Error handling is implemented for DuckDuckGo searches to ensure graceful failure
- The project uses Pixeltable's UDF (User Defined Functions) for search and prompt creation
