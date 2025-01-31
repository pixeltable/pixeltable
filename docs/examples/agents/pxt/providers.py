from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

import pixeltable as pxt


class Model(BaseModel):
    """A chat model that supports multiple providers"""
    provider: Literal['openai', 'anthropic'] = Field(..., description='The model provider')
    model_name: str = Field(..., description='Name of the model to use')
    temperature: float = Field(0.7, description='Temperature for response generation')
    max_tokens: Optional[int] = Field(None, description='Maximum tokens in response')


class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    def __init__(self, system_prompt: str, model_name: str, temperature: float, max_tokens: int):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def setup_columns(self, table: Any) -> None:
        """Set up the computed columns for the provider"""
        pass

    @abstractmethod
    def get_answer_column(self, table: Any) -> str:
        """Get the answer from the provider's response"""
        pass


class OpenAIProvider(ModelProvider):
    def setup_columns(self, table: Any) -> None:
        message_format = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': table.prompt},
        ]
        
        table.add_computed_column(
            openai_api_response=pxt.functions.openai.chat_completions(
                model=self.model_name,
                messages=message_format,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            if_exists='ignore',
        )

    def get_answer_column(self, table: Any) -> str:
        return table.openai_api_response.choices[0].message.content


class AnthropicProvider(ModelProvider):
    def setup_columns(self, table: Any) -> None:
        message_format = [{'role': 'user', 'content': table.prompt}]
        
        table.add_computed_column(
            anthropic_api_response=pxt.functions.anthropic.messages(
                messages=message_format,
                model=self.model_name,
                system=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            if_exists='ignore',
        )

    def get_answer_column(self, table: Any) -> str:
        return table.anthropic_api_response['content'][0]['text'] 