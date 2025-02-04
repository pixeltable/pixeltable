# Pixeltable Finance Agent Example

This example demonstrates how to create an AI agent that can retrieve and analyze stock market information using OpenAI's GPT model and the yfinance API.

## Overview

The example shows how to:
- Create a Pixeltable agent that can process financial queries
- Use yfinance as a tool to fetch real-time stock information
- Process queries through a multi-step pipeline using OpenAI's language model
- Get meaningful responses about stock information

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

- `create_agent.py`: Creates the finance agent with stock information tools and sets up the processing pipeline
- `chat.py`: Simple interface to interact with the finance agent
- `config.py`: Configuration settings (directory name and OpenAI model)
- `requirements.txt`: Required Python packages

## Usage

1. First, run the agent creation script:
```bash
python create_agent.py
```

2. Then use the chat interface to ask questions about stocks:
```bash
python chat.py
```

Example queries you can try:
- "What's the stock price of Nvidia?"
- "Tell me about Apple's market cap and PE ratio"
- "What's the current dividend yield for Microsoft?"

## How it Works

The agent processes queries through a multi-step pipeline:

1. **Initial Setup** (`create_agent.py`):
   - Creates a fresh Pixeltable directory
   - Defines a table to store queries and responses
   - Sets up the yfinance tool for fetching stock information

2. **Query Processing Pipeline**:
   - Takes user input query
   - Gets initial OpenAI response to determine which stock to look up
   - Invokes the yfinance tool to fetch real-time stock data
   - Sends the results back to OpenAI for final analysis
   - Returns a natural language response

3. **Chat Interface** (`chat.py`):
   - Provides a simple way to send queries to the agent
   - Displays the processed responses

## Configuration

The `config.py` file contains:
- `DIRECTORY`: The Pixeltable directory name for storing the agent
- `OPENAI_MODEL`: The OpenAI model to use for processing queries

## Dependencies

- pixeltable: For creating and managing the agent
- openai: For natural language processing
- yfinance: For fetching stock market data

## Note

Make sure you have your OpenAI API key properly configured in your environment before running the example. The agent uses OpenAI's GPT model to process queries and generate responses based on real-time stock data. 

```bash
export OPENAI_API_KEY=your_api_key
```

