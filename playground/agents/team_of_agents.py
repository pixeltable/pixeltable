# pip install pixeltable yfinance duckduckgo-search openai

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
    return stock.info


tools = pxt.tools(stock_info)

# Financial Agent
finance_agent = pxt.create_table(
    "agents.finance", {"prompt": pxt.String}, if_exists="ignore"
)
finance_agent_messages = [{"role": "user", "content": finance_agent.prompt}]

finance_agent.add_computed_column(
    initial_response=chat_completions(
        model="gpt-4o-mini", messages=finance_agent_messages, tools=tools
    )
)

finance_agent.add_computed_column(
    tool_output=invoke_tools(tools, finance_agent.initial_response)
)


# Portfolio Manager Agent
portfolio_manager = pxt.create_table(
    "agents.portfolio", {"prompt": pxt.String}, if_exists="ignore"
)
portfolio_manager_messages = [{"role": "user", "content": portfolio_manager.prompt}]
portfolio_manager.add_computed_column(
    initial_response=chat_completions(
        model="gpt-4o-mini", messages=portfolio_manager_messages, tools=tools
    )
)
portfolio_manager.add_computed_column(
    tool_output=invoke_tools(tools, portfolio_manager.initial_response)
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

portfolio_manager.add_computed_column(
    tool_response_prompt=create_prompt(portfolio_manager.prompt, portfolio_manager.tool_output)
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
