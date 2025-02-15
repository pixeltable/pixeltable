# pip install pixeltable yfinance openai

from typing import Optional

import pixeltable as pxt
from pixeltable.functions.openai import chat_completions, invoke_tools

import yfinance as yf

# Setup
pxt.drop_dir("agents", force=True)
pxt.create_dir("agents")

# Tool Definition
@pxt.udf
def stock_info(ticker: str) -> Optional[float]:
    """
    Get stock info for a given ticker symbol.

    Args:
        ticker - The ticker symbol of the stock to look up.
    """
    stock = yf.Ticker(ticker)
    return stock.info['currentPrice']


tools = pxt.tools(stock_info)

# Agent Table Creation and Initial Response
finance_agent = pxt.create_table(
    "agents.finance", {"prompt": pxt.String}, if_exists="ignore"
)
messages = [{"role": "user", "content": finance_agent.prompt}]

finance_agent.add_computed_column(
    initial_response=chat_completions(
        model="gpt-4o-mini", messages=messages, tools=tools
    ),
    if_exists="replace"
)

finance_agent.add_computed_column(
    tool_output=invoke_tools(tools, finance_agent.initial_response),
    if_exists="replace"
)


# Add tool response to prompt
@pxt.udf
def create_prompt(question: str, tool_outputs: list[dict]) -> str:
    return f"""
   QUESTION:

   {question}

   TOOL OUTPUTS:

   {tool_outputs}
  """


finance_agent.add_computed_column(
    tool_response_prompt=create_prompt(finance_agent.prompt, finance_agent.tool_output)
)

# Send back to OpenAI for final response
messages = [
    {
        "role": "system",
        "content": "Answer the users question based on the provided tool outputs.",
    },
    {"role": "user", "content": finance_agent.tool_response_prompt},
]

finance_agent.add_computed_column(
    final_response=chat_completions(model="gpt-4o-mini", messages=messages)
)

finance_agent.add_computed_column(
    answer=finance_agent.final_response.choices[0].message.content
)

# Usage Example
finance_agent.insert(prompt="What is the stock price of NVDA today?")
print(finance_agent.select(finance_agent.answer).collect())
