import config
from datetime import datetime
import pixeltable as pxt
from pixeltable.functions import openai
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Agent(BaseModel):
    """A persistent agent with optional index search capabilities and message history"""
    model_config = {"arbitrary_types_allowed": True}
    
    agent_name: str = Field(..., description="Name of the agent")
    system_prompt: str = Field(..., description="System prompt for the agent")
    agent_tools: Any = Field(None, description="Optional custom tools for the agent")
    index: Optional[pxt.Table] = Field(None, description="Optional pixeltable index containing data to search")
    purge: bool = Field(False, description="If True, delete existing agent history before creating")
    agent_table: Optional[pxt.Table] = Field(None, exclude=True)
    messages_table: Optional[pxt.Table] = Field(None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.agent_name = self.agent_name.replace(" ", "_")
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize or get existing agent tables"""
        # Delete agent tables if requested
        if self.purge:
            pxt.drop_table(f"{self.agent_name}_messages", if_not_exists='ignore', force=True)
            pxt.drop_table(self.agent_name, if_not_exists='ignore', force=True)

        # Create or get messages table
        messages_table_name = f"{self.agent_name}_messages"
        if messages_table_name not in pxt.list_tables():
            self.messages_table = pxt.create_table(
                path_str=messages_table_name,
                schema_or_df={
                    'role': pxt.String,  # 'user' or 'assistant'
                    'content': pxt.String,  # message content
                    'timestamp': pxt.Timestamp,  # for ordering
                },
                if_exists='ignore'
            )
        else:
            print(f"Messages table {messages_table_name} already exists")
            self.messages_table = pxt.get_table(messages_table_name)

        # Get existing or create new agent table
        if self.agent_name not in pxt.list_tables():
            # Create new table
            self.agent_table = pxt.create_table(
                path_str=self.agent_name, 
                schema_or_df={
                    'prompt': pxt.String,
                }, 
                if_exists='ignore'
            )
            
            # Use provided tools or create search tool if index exists
            tools = self.agent_tools
            if tools is None and self.index is not None:
                tools = create_search_tool(self.index)
            
            # Build table
            setup_agent_table(self.agent_table, tools, self.system_prompt, self.messages_table)
        else:
            print(f"Agent {self.agent_name} already exists")
            self.agent_table = pxt.get_table(self.agent_name)

    def batch_inject_messages(self, messages_to_inject: List[Dict]):
        """Inject multiple messages into the conversation history at once.
        
        Args:
            messages_to_inject: List of message dictionaries with 'role' and 'content' keys
        """
        # Validate message format
        for msg in messages_to_inject:
            if not all(key in msg for key in ['role', 'content']):
                raise ValueError("Each message must have 'role' and 'content' keys")
        
        # Insert all messages with current timestamp
        current_time = datetime.now()
        self.messages_table.insert([
            {'role': msg['role'], 'content': msg['content'], 'timestamp': current_time}
            for msg in messages_to_inject
        ])

    
    def get_messages(self):
        """Get the messages from the messages table.
        
        Returns:
            List of messages from the messages table
        """
        return pxt.get_table(f"{self.agent_name}_messages")

    def run(self, message: str, additional_context: Optional[List[Dict]] = None) -> str:
        """Run the agent with optional message injection.

        Args:
            message: The current user message
            additional_context: Optional list of messages to inject before processing the current message
            
        Returns:
            The agent's response
        """
        print(f"\n=== Running agent '{self.agent_name}' ===")
        print(f"Received message: {message}")
        

        # Get the latest tables
        agent_table = pxt.get_table(self.agent_name)
        messages_table = pxt.get_table(f"{self.agent_name}_messages")
        
        # Inject additional messages if provided
        if additional_context:
            print(f"Injecting {len(additional_context)} additional messages into conversation history")
            batch_inject_messages(messages_table, additional_context)
        else:
            print("No additional messages to inject")


        # Store user message in memory
        print("Storing user message in messages table...")
        messages_table.insert([{'role': 'user', 'content': message, 'timestamp': datetime.now()}])
        print("User message stored successfully")

        # Process through chat session
        print("Processing message through agent table...")
        agent_table.insert([{'prompt': message}])
        print("Message processed")

        # Get response
        print("Retrieving agent's response...")
        result = agent_table.select(agent_table.answer).tail(1)
        response = result['answer'][0]
        print(f"Retrieved response: {response}")

        # Store assistant response in memory
        print("Storing assistant response in messages table...")
        messages_table.insert([{'role': 'assistant', 'content': response, 'timestamp': datetime.now()}])
        print("Assistant response stored successfully")
        
        print("=== Agent run complete ===\n")
        return response

def create_search_tool(index=None):
    """Create a search tool that can be used by the agent.
    
    Args:
        index: Optional pixeltable index containing the data. If None, no search tool will be created.
        
    Returns:
        A pixeltable tool for searching data, or None if no index provided
    """
    if index is None:
        return None
        
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

# Create messages with history
@pxt.udf
def create_messages_with_history(past_context: List[Dict], system_prompt: str, current_prompt: str, injected_messages: List[Dict] = None) -> List[Dict]:
    """Create messages with history and optional injected messages.
    
    Args:
        past_context: Previous conversation history
        system_prompt: The system prompt
        current_prompt: Current user prompt
        injected_messages: Optional list of messages to inject before the current prompt
        
    Returns:
        List of messages including system prompt, history, injected messages, and current prompt
    """
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend([{'role': msg['role'], 'content': msg['content']} for msg in past_context])
    
    # Insert injected messages if provided
    if injected_messages:
        messages.extend(injected_messages)
        
    messages.append({'role': 'user', 'content': current_prompt})
    return messages

def setup_agent_table(agent_table: pxt.Table, agent_tools: pxt.tools, system_prompt: str, messages_table: pxt.Table):
    """Set up the agent table for tool use with message history.
    
    Args:
        agent_table: The pixeltable table to set up
        agent_tools: The tools available to the agent, or None if no tools
        system_prompt: The system prompt for the agent
        messages_table: Table storing conversation history
    """
    # Query to get recent message history
    @pxt.query
    def get_recent_messages():
        return (
            messages_table.order_by(messages_table.timestamp, asc=False)
            .select(role=messages_table.role, content=messages_table.content)
            .limit(10)
        )

    # Add message history column
    agent_table.add_computed_column(messages_context=get_recent_messages())

    # Add computed columns in sequence
    agent_table.add_computed_column(
        messages=create_messages_with_history(agent_table.messages_context, system_prompt, agent_table.prompt)
    )

    if agent_tools is not None:
        # Add tool-based response generation if tools are provided
        agent_table.add_computed_column(
            tool_response=openai.chat_completions(
                model=config.AGENT_MODEL,
                messages=agent_table.messages,
                tools=agent_tools,
            )
        )
        
        agent_table.add_computed_column(
            tool_result=openai.invoke_tools(agent_tools, agent_table.tool_response)
        )
        
        agent_table.add_computed_column(
            interpret_tool_result=create_prompt(agent_table.prompt, agent_table.tool_result)
        )

        # Set up final response with tool results
        result_messages = [
            {'role': 'system', 'content': "Answer the user's question from the tool result."},
            {'role': 'user', 'content': agent_table.interpret_tool_result},
        ]
    else:
        # Direct response generation without tools
        result_messages = agent_table.messages

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

def batch_inject_messages(messages_table, messages_to_inject: List[Dict]):
    """Inject multiple messages into the conversation history at once.
    
    Args:
        messages_to_inject: List of message dictionaries with 'role' and 'content' keys
    """
    # Validate message format
    for msg in messages_to_inject:
        if not all(key in msg for key in ['role', 'content']):
            raise ValueError("Each message must have 'role' and 'content' keys")
    
    # Insert all messages with current timestamp
    current_time = datetime.now()
    messages_table.insert([
        {'role': msg['role'], 'content': msg['content'], 'timestamp': current_time}
        for msg in messages_to_inject
    ])