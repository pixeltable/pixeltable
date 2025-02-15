# pip install phidata
from typing import Optional
import json

from phi.agent import Agent
from phi.model.openai import OpenAIChat

import yfinance as yf

def stock_info(ticker: str) -> Optional[float]:
    """
    Get stock info for a given ticker symbol.
    """
    stock = yf.Ticker(ticker)
    return stock.info

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[stock_info],
    show_tool_calls=True,
    markdown=True,
)

finance_agent.print_response("What is the current price of NVDA?", stream=True)