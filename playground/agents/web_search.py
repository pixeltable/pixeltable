# pip install pixeltable openai duckduckgo-search
import pixeltable as pxt
from pixeltable.functions.openai import chat_completions, invoke_tools
from duckduckgo_search import DDGS

# Setup
pxt.drop_dir("agents", force=True)
pxt.create_dir("agents")


# Tool Definition
@pxt.udf
def search_news(keywords: str, max_results: int = 20) -> str:
    """Search news using DuckDuckGo and return results."""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(
                keywords=keywords,
                region="wt-wt",
                safesearch="off",
                timelimit="m",
                max_results=max_results,
            )
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. Title: {r['title']}\n"
                    f"   Source: {r['source']}\n"
                    f"   Published: {r['date']}\n"
                    f"   Snippet: {r['body']}\n"
                )
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"


tools = pxt.tools(search_news)

# Agent Table Creation and Initial Response
news_agent = pxt.create_table("agents.news", {"prompt": pxt.String}, if_exists="ignore")
messages = [{"role": "user", "content": news_agent.prompt}]

news_agent.add_computed_column(
    initial_response=chat_completions(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice=tools.choice(required=True),
    )
)

news_agent.add_computed_column(
    tool_output=invoke_tools(tools, news_agent.initial_response)
)


# Add news results to prompt
@pxt.udf
def create_prompt(question: str, news_outputs: list[dict]) -> str:
    return f"""
    QUESTION:

    {question}

    NEWS RESULTS:

    {news_outputs}
    """


news_agent.add_computed_column(
    news_response_prompt=create_prompt(news_agent.prompt, news_agent.tool_output)
)

# Send back to OpenAI for final response
messages = [
    {
        "role": "system",
        "content": "Summarize the news results to answer the user's question.",
    },
    {"role": "user", "content": news_agent.news_response_prompt},
]

news_agent.add_computed_column(
    final_response=chat_completions(model="gpt-4o-mini", messages=messages)
)

news_agent.add_computed_column(
    answer=news_agent.final_response.choices[0].message.content
)

# Usage Example
news_agent.insert(prompt="What are the latest in Los Angeles?")
print(news_agent.select(news_agent.tool_output, news_agent.answer).collect())
