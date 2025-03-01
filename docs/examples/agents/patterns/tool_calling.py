from agent import Agent

import yfinance as yf

import pixeltable as pxt

from typing import Optional

@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

@pxt.udf
def get_edits(article: str) -> str:
    """Edit a news article for clarity and accuracy."""
    editor = Agent(
        agent_name="News Article Editor",
        system_prompt="Given a news article, edit it for clarity and accuracy.",
        purge=False
    )
    print("Inserting article into editor table...")
    response = editor.run(article)
    print("Article successfully inserted")
    return response

stock_analyst = Agent(
    agent_name="stock_analyst",
    system_prompt="You are a stock analyst, who can access yahoo finance data. Help the user with their stock analysis.",
    agent_tools=pxt.tools(stock_info),
    purge=False
)

writer = Agent(
    agent_name="News Article Writer",
    system_prompt="Given a stock report, write a news article about the stock. Once finished, use the editor tool to edit the article.",
    agent_tools=pxt.tools(get_edits),
    purge=False
)

# Agent Flow
stock_analyst.run("First outline a report structure for a stock briefing to CIO. This outline serves as a guide and will be used throughout our conversation.")
stock_analyst.run("Next fill in the report with my details: Jim Smith, 123 Main St, Anytown, USA, 12345, jim.smith@example.com, CFA, Works at Point72 as a portfolio manager.")
report = stock_analyst.run("Now fill in the report for Factset (Ticker: FDS). Use your tools to get the latest stock information.")

article = writer.run(
    message=f"Write a news article about the stock. Use your editor tool as needed. {report}",
    # additional_context=formatted_messages
)

print(article)


# Context Injection for the writer agent
# agent_messages = stock_analyst.get_messages()
# agent_messages = agent_messages.order_by(agent_messages.timestamp, asc=False).select(role=agent_messages.role, content=agent_messages.content).limit(10).collect()
# formatted_messages = [
#     {"role": msg['role'], "content": msg['content']}
#     for msg in agent_messages
# ]




