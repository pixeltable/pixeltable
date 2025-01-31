from config import DIRECTORY, OPENAI_MODEL
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

import pixeltable as pxt
from pixeltable.functions import openai

# Create helper function for RAG prompt
@pxt.udf
def create_rag_prompt(question: str,  index_content: list[dict]) -> str:
    return f"""

    QUESTION:
    {question}
    
    RESULT:
    {index_content}
    """

class Agent(BaseModel):
    """A generic agent class for RAG-based interactions"""
    agent_name: str = Field(..., description="Name of the agent and its environment")
    system_prompt: str = Field(..., description="System prompt for the agent")
    knowledge: Optional[str] = Field(None, description="Name of the knowledge table containing embeddings")
    persist: bool = Field(True, description="Whether to persist between runs")
    
    # Private fields with default values
    agent_table: Optional[Any] = Field(None, exclude=True)
    agent_tools: Optional[Any] = Field(None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize the agent after validation"""
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Set up the agent environment and table"""
        if not self.persist:
            # Start fresh each time
            pxt.create_environment(name=self.agent_name, if_exists='ignore')

        # Create agent table
        pxt.create_table(
            path_str=f'{DIRECTORY}.{self.agent_name}',
            schema_or_df={'prompt': pxt.String},
            if_exists='ignore',
        )

        # Get agent table
        self.agent_table = pxt.get_table(f'{DIRECTORY}.conversations')

        if self.knowledge:
            self._setup_knowledge_tools()

        self._setup_table_columns()

    def _setup_knowledge_tools(self) -> None:
        """Set up knowledge-based search tools"""
        # Get index
        embedding_index = pxt.get_table(self.knowledge)

        # Create a tool to search for relevant passages
        @pxt.query
        def search(query_text: str) -> str:
            """
            Search tool to find relevant passages.
            Args:
                query_text: The search query
            Returns:
                Top 10 most relevant passages
            """
            similarity = embedding_index.text.similarity(query_text)
            return (
                embedding_index.order_by(similarity, asc=False)
                .select(embedding_index.text, sim=similarity)
                .limit(10)
            )

        self.agent_tools = pxt.tools(search)

    def _setup_table_columns(self) -> None:
        """Set up the computed columns for the agent table"""
        # Define message structure to create tool response
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.agent_table.prompt},
        ]

        # Add tool response column
        self.agent_table.add_computed_column(
            tool_response=openai.chat_completions(
                model=OPENAI_MODEL,
                messages=messages,
                tools=self.agent_tools,
                tool_choice=self.agent_tools.choice(required=True) if self.agent_tools else None,
            ),
            if_exists='ignore'
        )

        # Add tool execution column
        self.agent_table.add_computed_column(
            tool_result=openai.invoke_tools(self.agent_tools, self.agent_table.tool_response) if self.agent_tools else None,
            if_exists='ignore'
        )

        # Add interpreted result column
        self.agent_table.add_computed_column(
            interpret_tool_result=create_rag_prompt(self.agent_table.prompt, self.agent_table.tool_result),
            if_exists='ignore'
        )

        # Set up final response
        tool_result_message = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.agent_table.interpret_tool_result},
        ]

        # Add final response column
        self.agent_table.add_computed_column(
            final_response=openai.chat_completions(
                model=OPENAI_MODEL,
                messages=tool_result_message
            ),
            if_exists='ignore'
        )

        # Add answer column
        self.agent_table.add_computed_column(
            answer=self.agent_table.final_response.choices[0].message.content,
            if_exists='ignore'
        )

    def get_table(self) -> Any:
        """Get the agent table"""
        return self.agent_table
