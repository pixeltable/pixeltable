from typing import Dict, List
import pixeltable as pxt
from pixeltable.functions import openai
from tools import web_search_and_ingest, get_stock_data, search_news, search_documents
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
def create_summary_messages(input_query: str, results: Dict) -> List[Dict[str, str]]:
    return [{
        'role': 'system',
        'content': get_summary_prompt(results)
    }, {
        'role': 'user',
        'content': input_query
    }]

def create_research_table():
    """Create and configure research table with computed columns."""
    # Initialize embedding index
    setup_embedding_index()

    # Create base table
    research_table = pxt.create_table(
        'research.queries',
        {
            'input': pxt.String
        }
    )

    # Set up tools
    tools = pxt.tools(
        web_search_and_ingest,
        get_stock_data,
        search_news,
        search_documents
    )

    # Initial query analysis
    research_table.add_computed_column(
        messages=create_messages(research_table.input)
    )

    # Get LLM response with tools
    research_table.add_computed_column(
        response=openai.chat_completions(
            model='gpt-4o-mini',
            messages=research_table.messages,
            tools=tools,
            max_tokens=1000
        )
    )

    # Execute tools based on LLM response
    research_table.add_computed_column(
        tool_results=openai.invoke_tools(tools, research_table.response)
    )

    # Extract initial answer
    research_table.add_computed_column(
        initial_answer=research_table.response.choices[0].message.content
    )

    # Create messages for summary
    research_table.add_computed_column(
        summary_messages=create_summary_messages(
            research_table.input,  # Pass input query
            research_table.tool_results  # Pass tool results
        )
    )

    research_table.add_computed_column(news_results=research_table.tool_results['search_news'])
    research_table.add_computed_column(stock_results=research_table.tool_results['get_stock_data'])
    research_table.add_computed_column(doc_results=research_table.tool_results['search_documents']) 
    research_table.add_computed_column(web_results=research_table.tool_results['web_search_and_ingest'])

    # Generate final summary
    research_table.add_computed_column(
        final_summary=openai.chat_completions(
            messages=research_table.summary_messages,
            model='gpt-4o-mini',
            temperature=0.3,
            max_tokens=1000
        ).choices[0].message.content
    )

    return research_table
