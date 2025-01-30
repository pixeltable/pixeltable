from config import ANTHROPIC_MODEL, DIRECTORY
from duckduckgo_search import DDGS

import pixeltable as pxt
from pixeltable.functions.anthropic import invoke_tools, messages

# Create fresh environment
pxt.drop_dir(DIRECTORY, force=True)
pxt.create_dir(DIRECTORY, if_exists='ignore')

# Create Agent Table
news_agent = pxt.create_table(f'{DIRECTORY}.news', {'prompt': pxt.String}, if_exists='ignore')


# Add duckduckgo search tool
@pxt.udf
def search_news(keywords: str, max_results: int = 20) -> str:
    """Search news using DuckDuckGo and return results."""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(
                keywords=keywords,
                region='wt-wt',
                safesearch='off',
                timelimit='m',
                max_results=max_results,
            )
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f'{i}. Title: {r["title"]}\n'
                    f'   Source: {r["source"]}\n'
                    f'   Published: {r["date"]}\n'
                    f'   Snippet: {r["body"]}\n'
                )
            return '\n'.join(formatted_results)
    except Exception as e:
        return f'Search failed: {str(e)}'


tools = pxt.tools(search_news)

# Add initial response from OpenAI
news_agent.add_computed_column(
    initial_response=messages(
        model=ANTHROPIC_MODEL,
        messages=[{'role': 'user', 'content': news_agent.prompt}],
        tools=tools,
        tool_choice=tools.choice(required=True),
    )
)

# Invoke tools
news_agent.add_computed_column(tool_output=invoke_tools(tools, news_agent.initial_response))


# Add invoked results to prompt
@pxt.udf
def create_prompt(question: str, news_outputs: list[dict]) -> str:
    return f"""
    QUESTION:

    {question}

    RESULTS:

    {news_outputs}
    """


# Create prompt with invoked response
news_agent.add_computed_column(news_response_prompt=create_prompt(news_agent.prompt, news_agent.tool_output))

# Send back to Anthropic for final response
news_agent.add_computed_column(
    final_response=messages(
        model=ANTHROPIC_MODEL,
        messages=[{'role': 'user', 'content': news_agent.news_response_prompt}],
        system="Answer the user's question based on the results.",
    )
)

news_agent.add_computed_column(answer=news_agent.final_response.content[0].text)
