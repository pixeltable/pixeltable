# Pixeltable Finance Agent Example

This example demonstrates how to create an AI agent that can retrieve and analyze stock market information using Anthropic's Claude model and the yfinance API.

## Overview

The example shows how to:
- Create an agent that can process financial queries
- Use yfinance as a tool to fetch real-time stock information
- Process the results using Anthropic's Claude model
- Get meaningful responses about stock information

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

The following dependencies are needed:
- pixeltable
- anthropic
- yfinance

## Configuration

The project configuration is stored in `config.py`:
- `DIRECTORY`: The Pixeltable directory where data will be stored
- `ANTHROPIC_MODEL`: The Anthropic model to use for processing

## Setup and Usage

1. First, run the agent creation script:
```bash
python create_agent.py
```

This script will:
- Create a finance agent table
- Add the yfinance stock information tool
- Set up the processing pipeline with Claude

2. To interact with the agent, use `chat.py`:
```bash
python chat.py
```

The agent will:
1. Take your finance-related query
2. Fetch relevant stock information using yfinance
3. Process the results using Claude
4. Return a detailed response

## Example Usage

```python
import pixeltable as pxt
from config import DIRECTORY

finance_agent = pxt.get_table(f"{DIRECTORY}.finance")
finance_agent.insert(prompt="What's the stock price of Nvidia?")
```

## Project Structure

- `create_agent.py`: Creates and configures the finance agent
- `chat.py`: Interface for interacting with the agent
- `config.py`: Project configuration
- `requirements.txt`: Package dependencies

## Notes

- The agent uses yfinance to fetch real-time stock information
- It can process various types of financial queries about stocks
- Make sure you have your Anthropic API key properly configured in your environment before running the example:

```bash
export ANTHROPIC_API_KEY=your_api_key
```

