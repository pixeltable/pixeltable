import config

import pixeltable as pxt
from pixeltable.functions import openai

def create_search_tool(index):
    """Create a PDF search tool that can be used by the agent.
    
    Args:
        index: The pixeltable index containing the PDF data
        
    Returns:
        A pixeltable tool for searching PDF content
    """
    @pxt.query
    def search_pdf(query_text: str) -> str:
        """Search tool to find relevant pdf passages.
        
        Args:
            query_text: The search query
        Returns:
            Top 10 most relevant passages
        """
        similarity = index.text.similarity(query_text)
        return (
            index.order_by(similarity, asc=False)
            .select(index.text, sim=similarity)
            .limit(10)
        )
    
    return pxt.tools(search_pdf)

def setup_agent_table(agent_table, agent_tools, system_prompt: str):
    """Set up the agent table for tool use.
    
    Args:
        agent_table: The pixeltable table to set up
        agent_tools: The tools available to the agent
        system_prompt: The system prompt for the agent
    """
    # Define message structure for tool response
    tool_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': agent_table.prompt},
    ]

    # Add computed columns in sequence
    agent_table.add_computed_column(
        tool_response=openai.chat_completions(
            model=config.AGENT_MODEL,
            messages=tool_messages,
            tools=agent_tools,
            tool_choice=agent_tools.choice(required=True),
        )
    )
    
    agent_table.add_computed_column(
        tool_result=openai.invoke_tools(agent_tools, agent_table.tool_response)
    )
    
    agent_table.add_computed_column(
        interpret_tool_result=create_prompt(agent_table.prompt, agent_table.tool_result)
    )

    # Set up final response
    result_messages = [
        {'role': 'system', 'content': "Answer the user's question from the tool result."},
        {'role': 'user', 'content': agent_table.interpret_tool_result},
    ]

    agent_table.add_computed_column(
        final_response=openai.chat_completions(
            model=config.AGENT_MODEL, 
            messages=result_messages
        )
    )
    
    agent_table.add_computed_column(
        answer=agent_table.final_response.choices[0].message.content
    )

@pxt.udf
def create_prompt(question: str, tool_result: list[dict]) -> str:
    """Create a prompt combining the question and search results.
    
    Args:
        question: The user's question
        tool_result: Results from the search tool
        
    Returns:
        A formatted prompt string
    """
    return f"""
    QUESTION:
    {question}
    
    SEARCH TOOL RESULT:
    {tool_result}
    """

def create_agent(agent_name: str, index: pxt.Table, system_prompt: str, purge: bool = False) -> pxt.Table:
    """Create or get a persistent agent with index search capabilities.
    
    Args:
        index: The pixeltable index containing data to search
        purge: If True, delete existing agent historybefore creating
        
    Returns:
        A pixeltable Table configured for the agent
    """
    # Delete agent table if requested
    if purge:
        pxt.drop_table(agent_name, force=True)

    # Get existing or create new agent table
    if agent_name not in pxt.list_tables():
        # Create new table
        agent = pxt.create_table(
            path_str=agent_name, 
            schema_or_df={'prompt': pxt.String}, 
            if_exists='ignore'
        )
        
        # Create and set up tools
        agent_tools = create_search_tool(index)
        
        # Build table
        setup_agent_table(agent, agent_tools, system_prompt)
    else:
        agent = pxt.get_table(agent_name)

    return agent
