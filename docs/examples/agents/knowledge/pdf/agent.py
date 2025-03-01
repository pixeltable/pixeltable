import pixeltable as pxt
from pixeltable.functions import openai

def create_agent(
    agent_name: str, 
    index: pxt.Table, 
    llm_model_name: str, 
    system_prompt: str, 
    reset_history: bool = False) -> pxt.Table:

    """
    Create or get a persistent agent with index search capabilities.
    
    Args:
        index: The pixeltable index containing data to search
        reset_history: If True, delete existing agent historybefore creating
        
    Returns:
        A pixeltable Table configured for the agent
    """

    # Delete agent table if requested
    if reset_history:
        pxt.drop_table(agent_name, force=True)

    # Get existing or create new agent table
    if agent_name not in pxt.list_tables():
        # Create new table
        agent = pxt.create_table(
            path_str=agent_name, 
            schema_or_df={'prompt': pxt.String}, 
            if_exists='ignore'
        )
        
        # Add search tool
        agent_tools = create_search_tool(index)
        
        # Add tool calling to agent table
        add_tool_calling(agent, llm_model_name, agent_tools, system_prompt)

def create_search_tool(index: pxt.Table) -> pxt.tools:
    """Create a search tool that can be used by the agent.
    
    Args:
        index: The pixeltable index containing the data
        
    Returns:
        A pixeltable tool for searching data
    """
    @pxt.query
    def search(query_text: str) -> str:
        """Search tool to find relevant passages.
        
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
    
    return pxt.tools(search)

def add_tool_calling(
    agent_table: pxt.Table, 
    llm_model_name: str, 
    agent_tools: pxt.tools, 
    system_prompt: str):

    """Set up the agent table for tool use.
    
    Args:
        agent_table: The pixeltable table to set up
        llm_model_name: The LLM model name
        agent_tools: The tools available to the agent
        system_prompt: The system prompt for the agent
    """

    tool_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': agent_table.prompt},
    ]

    agent_table.add_computed_column(
        tool_response=openai.chat_completions(
            model=llm_model_name,
            messages=tool_messages,
            tools=agent_tools,
            tool_choice=agent_tools.choice(required=True),
        )
    )
    
    agent_table.add_computed_column(
        tool_result=openai.invoke_tools(agent_tools, agent_table.tool_response)
    )
    
    agent_table.add_computed_column(
        interpret_tool_result=create_tool_prompt(agent_table.prompt, agent_table.tool_result)
    )

    result_messages = [
        {'role': 'system', 'content': "Answer the user's question from the tool result."},
        {'role': 'user', 'content': agent_table.interpret_tool_result},
    ]

    agent_table.add_computed_column(
        final_response=openai.chat_completions(
            model=llm_model_name, 
            messages=result_messages
        )
    )
    
    agent_table.add_computed_column(
        answer=agent_table.final_response.choices[0].message.content
    )

@pxt.udf
def create_tool_prompt(question: str, tool_result: list[dict]) -> str:
    return f"""
    QUESTION:
    {question}
    
    SEARCH TOOL RESULT:
    {tool_result}
    """