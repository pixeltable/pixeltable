# Pixeltable Finance Agent Example

This example demonstrates how to create an AI agent that can retrieve and analyze stock market information using OpenAI's GPT model and the yfinance API.

## Overview

The example shows how to:
- Set up a Pixeltable environment for the finance agent
- Create an agent that can process financial queries
- Use yfinance as a tool to fetch real-time stock information
- Process the results using OpenAI's language model
- Get meaningful responses about stock information

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Files Structure

- `01_setup.py`: Initializes the Pixeltable environment
- `02_create_agent.py`: Creates the finance agent with stock information tools
- `chat.py`: Example usage of the finance agent
- `config.py`: Configuration settings
- `requirements.txt`: Required Python packages

## Usage

1. First, run the setup script:
```bash
python 01_setup.py
```

2. You can then use the chat interface to ask questions about stocks:
```bash
python chat.py
```

Example query:
```python
"What's the stock price of Nvidia?"
```

## How it Works

1. The setup script (`01_setup.py`) creates a fresh Pixeltable environment.

2. The agent creation script (`02_create_agent.py`):
   - Creates a table to store queries and responses
   - Defines a stock_info tool using yfinance
   - Processes queries through multiple steps:
     - Initial OpenAI response
     - Tool invocation
     - Final response generation

3. The chat interface (`chat.py`) provides a simple way to interact with the agent.

## Configuration

The `config.py` file contains:
- `DIRECTORY`: The Pixeltable directory name
- `OPENAI_MODEL`: The OpenAI model to use

## Dependencies

- pixeltable
- openai
- yfinance

## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example. 

```bash
export OPENAI_API_KEY=your_api_key
```

