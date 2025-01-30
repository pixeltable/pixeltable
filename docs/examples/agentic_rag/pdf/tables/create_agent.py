from config import DIRECTORY, OPENAI_MODEL

import pixeltable as pxt
from pixeltable.functions import openai

# Create agent table
agent_table = pxt.create_table(
    path_str=f'{DIRECTORY}.conversations',
    schema_or_df={'prompt': pxt.String},
    if_exists='ignore',
)

# Get pdf index
pdf_embedding_index = pxt.get_table(f'{DIRECTORY}.pdf_chunks')


# Create query-as-a-tool for Agentic RAG
@pxt.query
def search_pdf(query_text: str) -> str:
    """
    Search tool to find relevant pdf passages.
    Args:
        query_text: The search query
    Returns:
        Top 10 most relevant passages
    """
    similarity = pdf_embedding_index.text.similarity(query_text)
    return (
        pdf_embedding_index.order_by(similarity, asc=False).select(pdf_embedding_index.text, sim=similarity).limit(10)
    )


agent_tools = pxt.tools(search_pdf)


# Create prompt for Agentic RAG
@pxt.udf
def create_prompt(question: str, tool_result: list[dict]) -> str:
    return f"""

    QUESTION:
    {question}
    
    TOOL RESULT:
    {tool_result}
    """


# Define message structure to create tool response
messages = [
    {'role': 'system', 'content': "Create a search query based on the user's question."},
    {'role': 'user', 'content': agent_table.prompt},
]

# Add tool response column
agent_table.add_computed_column(
    tool_response=openai.chat_completions(
        model=OPENAI_MODEL,
        messages=messages,
        tools=agent_tools,
        tool_choice=agent_tools.choice(required=True),
    )
)

# Add tool execution column
agent_table.add_computed_column(tool_result=openai.invoke_tools(agent_tools, agent_table.tool_response))

# Add interpreted result column
agent_table.add_computed_column(interpret_tool_result=create_prompt(agent_table.prompt, agent_table.tool_result))

# Set up final response
tool_result_message = [
    {'role': 'system', 'content': "Answer the user's question from the tool result."},
    {'role': 'user', 'content': agent_table.interpret_tool_result},
]

# Add final response column
agent_table.add_computed_column(
    final_response=openai.chat_completions(model=OPENAI_MODEL, messages=tool_result_message)
)

# Add answer column
agent_table.add_computed_column(answer=agent_table.final_response.choices[0].message.content)
