import uuid
from datetime import datetime
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

import pixeltable as pxt


class AgentTable(BaseModel):
    """Schema definitions for the agent table"""

    # Message fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description='Unique message ID')
    timestamp: datetime = Field(default_factory=datetime.now, description='Message timestamp')
    prompt: str = Field(..., description="User's input message")
    
    # Agent configuration fields
    provider: Literal['openai', 'anthropic'] = Field(..., description='The model provider')
    model_name: str = Field(..., description='Name of the model to use')
    system_prompt: str = Field(..., description='System prompt for the agent')
    agent_name: str = Field(..., description='Name of the agent')

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Get the pixeltable schema definition"""
        return {
            'id': pxt.String,
            'timestamp': pxt.Timestamp,
            'prompt': pxt.String,
            'provider': pxt.String,
            'model_name': pxt.String, 
            'system_prompt': pxt.String,
            'agent_name': pxt.String,
        }
