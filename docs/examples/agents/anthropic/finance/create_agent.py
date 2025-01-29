from typing import Optional

import yfinance as yf
from config import DIRECTORY, ANTHROPIC_MODEL

import pixeltable as pxt
from pixeltable.functions.anthropic import messages, invoke_tools

# Create Agent Table
finance_agent = pxt.create_table(f'{DIRECTORY}.finance', {'prompt': pxt.String}, if_exists='ignore')

# Add yfinance tool
@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


tools = pxt.tools(stock_info)

# Add initial response from OpenAI
finance_agent.add_computed_column(
    initial_response=messages(
        model=ANTHROPIC_MODEL,
        messages=[{'role': 'user', 'content': finance_agent.prompt}],
        tools=tools,
        tool_choice=tools.choice(required=True),
    )
)

# Invoke tools
finance_agent.add_computed_column(tool_output=invoke_tools(tools, finance_agent.initial_response))


# Add invoked results to prompt
@pxt.udf
def create_prompt(question: str, tool_outputs: list[dict]) -> str:
    return f"""
    QUESTION:

    {question}

    RESULTS:

    {tool_outputs}
    """


# Create prompt with invoked response
finance_agent.add_computed_column(stock_response_prompt=create_prompt(finance_agent.prompt, finance_agent.tool_output))

# Send back to Anthropic for final response
finance_agent.add_computed_column(
    final_response=messages(
        model=ANTHROPIC_MODEL,
        messages=[{'role': 'user', 'content': finance_agent.stock_response_prompt}],
        system="Answer the user's question based on the results.",
    )
)

finance_agent.add_computed_column(answer=finance_agent.final_response.content[0].text)
