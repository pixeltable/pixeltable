import pixeltable as pxt
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the Agent"""

    context_window_size: int = 5
    model_name: str = "gpt-4o-mini"
    system_prompt: str = """You are a chatbot with memory capabilities. 
    Use the conversation history to provide contextual and informed responses.
    Remember previous interactions and refer back to them when relevant."""


class Agent:
    def __init__(self, agent_id: str, config: Optional[AgentConfig] = None):
        """Initialize the Agent with its own memory space and configuration"""
        self.agent_id = agent_id
        self.config = config or AgentConfig()

        # Initialize database tables
        self._setup_database()

    def _setup_database(self) -> None:
        """Setup the database tables for the agent"""
        # Create a unique namespace for this agent
        self.db_namespace = f"chatbot_{self.agent_id}"
        pxt.drop_dir(self.db_namespace, force=True)
        pxt.create_dir(self.db_namespace)

        # Create memory table
        self.memory = pxt.create_table(
            f"{self.db_namespace}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
        )

        # Query to get recent memory
        @self.memory.query
        def get_recent_memory():
            return (
                self.memory.order_by(self.memory.timestamp.desc())
                .select(role=self.memory.role, content=self.memory.content)
                .limit(self.config.context_window_size)
            )

    def _create_messages(
        self, past_context: List[Dict], current_message: str
    ) -> List[Dict]:
        """Create messages list with system prompt, memory context and new message"""
        messages = [{"role": "system", "content": self.config.system_prompt}]

        messages.extend(
            [{"role": msg["role"], "content": msg["content"]} for msg in past_context]
        )

        messages.append({"role": "user", "content": current_message})

        return messages

    def _get_llm_response(self, messages: List[Dict]) -> str:
        """Get response from the language model"""
        response = pxt.functions.openai.chat_completions(
            messages=messages, model=self.config.model_name
        )
        return response.choices[0].message.content

    def chat(self, message: str) -> str:
        """Process a message and return the response"""
        try:
            # Store user message in memory
            self.memory.insert(
                [{"role": "user", "content": message, "timestamp": datetime.now()}]
            )

            # Get recent memory
            context = (
                self.memory.queries.get_recent_memory().collect().to_dict("records")
            )

            # Create messages for LLM
            messages = self._create_messages(context, message)

            # Get response from LLM
            response = self._get_llm_response(messages)

            # Store assistant response in memory
            self.memory.insert(
                [
                    {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now(),
                    }
                ]
            )

            return response

        except Exception as e:
            return f"Error: {str(e)}"

    def get_memory(self) -> List[Dict]:
        """Retrieve all memories for this agent"""
        return (
            self.memory.select(
                role=self.memory.role,
                content=self.memory.content,
                timestamp=self.memory.timestamp,
            )
            .order_by(self.memory.timestamp)
            .collect()
            .to_dict("records")
        )

    def clear_memory(self) -> None:
        """Clear all memories for this agent"""
        self.memory.delete()


def main():
    # Create an agent with custom configuration
    config = AgentConfig(
        context_window_size=3,
        model_name="gpt-4-mini",
        system_prompt="You are a friendly and helpful assistant with memory capabilities.",
    )
    agent = Agent("demo_agent", config)

    print("Bot: Hello! I'm a chatbot with memory. Let's talk!")

    # Example conversation
    messages = [
        "Hi! My name is Alice.",
        "What's the weather like today?",
        "Thanks! Can you remember my name?",
        "What was the first thing I asked you about?",
    ]

    for i, message in enumerate(messages, 1):
        print(f"\nUser: {message}")
        response = agent.chat(message)
        print(f"Bot: {response}")

    # Example of accessing memory
    print("\nMemory contents:")
    for memory in agent.get_memory():
        print(f"{memory['timestamp']}: {memory['role']} - {memory['content']}")


if __name__ == "__main__":
    main()
