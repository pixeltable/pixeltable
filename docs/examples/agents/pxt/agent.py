from typing import Any, Optional

from pydantic import BaseModel, Field

import pixeltable as pxt

from .types import AgentTable
from .providers import Model, ModelProvider, OpenAIProvider, AnthropicProvider


class Agent(BaseModel):
    model: Model = Field(..., description='The chat model to use')
    agent_name: str = Field(..., description='Name of the agent and its environment')
    system_prompt: str = Field(..., description='System prompt for the agent')
    clear_cache: bool = Field(False, description='Whether to recreate the agent from scratch')

    # Private fields with default values
    agent_table: Optional[Any] = Field(None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize the agent after validation"""
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Set up the agent environment and table"""
        table_path = f'{self.agent_name.replace(" ", "_")}'

        if self.clear_cache is True:
            pxt.drop_table(table_path, force=True)

        if table_path not in pxt.list_tables():
            pxt.create_table(path_str=table_path, schema_or_df=AgentTable.get_schema())

        self.agent_table = pxt.get_table(table_path)

        self._setup_table_columns()

    def _get_provider(self) -> ModelProvider:
        """Get the appropriate model provider"""
        providers = {
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
            # Add more providers here as needed
        }
        
        provider_class = providers.get(self.model.provider)
        if not provider_class:
            raise ValueError(f"Unsupported model provider: {self.model.provider}")
            
        return provider_class(
            system_prompt=self.system_prompt,
            model_name=self.model.model_name,
            temperature=self.model.temperature,
            max_tokens=self.model.max_tokens
        )

    def _setup_table_columns(self) -> None:
        """Set up the computed columns for the agent table"""
        provider = self._get_provider()
        
        # Set up provider-specific columns
        provider.setup_columns(self.agent_table)
        
        # Add answer column
        self.agent_table.add_computed_column(
            answer=provider.get_answer_column(self.agent_table),
            if_exists='ignore'
        )

    def get_history(self) -> Any:
        """Get the agent table"""
        return self.agent_table

    def run(self, prompt: str) -> str:
        """Run the agent with a prompt and return the answer"""
        # Create a new record in the agent table
        record = AgentTable(
            prompt=prompt,
            provider=self.model.provider,
            model_name=self.model.model_name,
            system_prompt=self.system_prompt,
            agent_name=self.agent_name
        )
        self.agent_table.insert([{
            'id': record.id,
            'timestamp': record.timestamp,
            'prompt': record.prompt,
            'provider': record.provider,
            'model_name': record.model_name,
            'system_prompt': record.system_prompt,
            'agent_name': record.agent_name
        }])

        # Get the latest result
        result = self.agent_table.select(self.agent_table.answer).collect()[-1]

        # Return the answer
        return result['answer']
