import pixeltable as pxt
from agent import Agent, run
import yfinance as yf
from typing import Optional

@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

@pxt.udf
def get_stock_history(ticker: str, period: str = "1mo") -> dict:
    """Get historical stock data for a given ticker symbol.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return {
        "latest_price": history['Close'][-1],
        "highest_price": history['High'].max(),
        "lowest_price": history['Low'].min(),
        "average_volume": history['Volume'].mean()
    }

# Create tools from our UDFs
tools = pxt.tools(stock_info, get_stock_history)

gavin_agent = Agent(
    agent_name="Financial_Researcher",
    system_prompt="""You are a stock analyst. You have access to two tools:
    1. stock_info: Gets detailed information about a stock
    2. get_stock_history: Gets historical price and volume data for a stock
    
    Use these tools to provide detailed analysis when asked about stocks.""",
    agent_tools=tools,
    purge=True  # Start fresh
)

phil_agent = Agent(
    agent_name="Phil_the_critic",
    system_prompt="Provide feedback on the assistant's response. Ensure it answers the users question. Be very critical. If there is no critique, respond with <OK>",
    purge=True  # Start fresh
)

# Get the financial researcher table
gavin_agent = pxt.get_table("Financial_Researcher")

run(
    agent_name="Financial_Researcher",
    message="create a 100 word report on NVDIA.",
)

max_iterations = 10
iteration = 0
while iteration < max_iterations:
    agent_messages = pxt.get_table("Financial_Researcher_messages")
    additional_context = (
        agent_messages
        .select(role=agent_messages.role, content=agent_messages.content)
        .tail(100) # last 100 messages
    )
    formatted_context = [{'role': msg['role'], 'content': msg['content']} for msg in additional_context]
    print(f"\nIteration {iteration + 1}:")
    print("Current context:", formatted_context)

    reflection_response = run(
        agent_name="Phil_the_critic",
        message="Provide feedback on the assistant's response.",
        injected_messages=formatted_context
    )

    print("Critic response:", reflection_response)
    if "<OK>" not in reflection_response:
        gavin_agent.insert([{'prompt': reflection_response}])
        print("Writer's new response:", gavin_agent.select(gavin_agent.answer).tail(1)['answer'][0])
        iteration += 1
    else:
        print(f"No critique, using the last response: {formatted_context[-1]['content']}")
        canidate_response = formatted_context[-1]['content']
        break

print(f"\nFinal response after {iteration + 1} iterations:")
print(canidate_response)
