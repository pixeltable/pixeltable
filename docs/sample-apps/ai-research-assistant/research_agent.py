from typing import Dict, List
import pixeltable as pxt
from pixeltable.functions import openai
from tools import get_stock_data, search_news, search_documents
from embeddings import setup_embedding_index
from prompts import get_research_prompt, get_summary_prompt

@pxt.udf
def create_messages(input_query: str) -> List[Dict[str, str]]:
    return [{
        'role': 'system',
        'content': get_research_prompt(input_query)
    }, {
        'role': 'user',
        'content': input_query
    }]

@pxt.udf
def create_summary_messages(tool_results: Dict) -> List[Dict[str, str]]:
    return [{
        'role': 'system',
        'content': get_summary_prompt(tool_results)
    }]

def create_research_table():
    """Create and configure research table with computed columns."""
    setup_embedding_index()

    research_table = pxt.create_table(
        'research.queries',
        {
            'input': pxt.String
        }
    )

    tools = pxt.tools(
        get_stock_data,
        search_news,
        search_documents
    )

    # Add computed columns
    research_table.add_computed_column(messages=create_messages(research_table.input))
    research_table.add_computed_column(
        response=openai.chat_completions(
            model='gpt-4o-mini',
            messages=research_table.messages,
            tools=tools,
            max_tokens=1000
        )
    )
    research_table.add_computed_column(tool_results=openai.invoke_tools(tools, research_table.response))
    research_table.add_computed_column(final_answer=research_table.response.choices[0].message.content)
    research_table.add_computed_column(summary_messages=create_summary_messages(research_table.tool_results))
    research_table.add_computed_column(
        final_summary=openai.chat_completions(
            messages=research_table.summary_messages,
            model='gpt-4o-mini',
            temperature=0.3,
            max_tokens=1000
        ).choices[0].message.content
    )

    return research_table