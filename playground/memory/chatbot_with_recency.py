import pixeltable as pxt
from datetime import datetime
from typing import List, Dict

# Initialize Pixeltable
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")

# Create memory table to store conversation history
memory = pxt.create_table(
    path_str="chatbot.memory",
    schema_or_df={
        "role": pxt.String,  # 'user' or 'assistant'
        "content": pxt.String,  # message content
        "timestamp": pxt.Timestamp,  # for ordering and context window
    },
    if_exists="ignore",
)

# Create chat_session table for processing responses with context
chat_session = pxt.create_table(
    path_str="chatbot.chat_session",
    schema_or_df={"user_message": pxt.String, "timestamp": pxt.Timestamp},
    if_exists="ignore",
)


# Query to retrieve recent memory
@pxt.query
def get_recent_memory():
    return (
        memory.order_by(memory.timestamp)
        .select(role=memory.role, content=memory.content)
        .limit(10)
    )  # Get last 10 messages for context window


@pxt.udf
def create_messages(past_context: List[Dict], current_message: str) -> List[Dict]:
    """Create messages list with system prompt, memory context and new message"""
    messages = [
        {
            "role": "system",
            "content": """You are a chatbot with memory capabilities. 
        Use the conversation history to provide contextual and informed responses.
        Remember previous interactions and refer back to them when relevant.""",
        }
    ]

    # Add conversation history from memory
    messages.extend(
        [{"role": msg["role"], "content": msg["content"]} for msg in past_context]
    )

    # Add current message
    messages.append({"role": "user", "content": current_message})

    return messages


# Add computed columns for response generation pipeline
chat_session.add_computed_column(memory_context=get_recent_memory())
chat_session.add_computed_column(
    prompt=create_messages(chat_session.memory_context, chat_session.user_message)
)
chat_session.add_computed_column(
    llm_response=pxt.functions.openai.chat_completions(
        messages=chat_session.prompt, model="gpt-4o-mini"
    )
)
chat_session.add_computed_column(
    assistant_response=chat_session.llm_response.choices[0].message.content
)

############################################################
# Chatbot with Recent Memory
############################################################


def chat(message: str) -> str:
    """Process a message through the memory-enabled chatbot"""
    # Store user message in memory
    memory.insert([{"role": "user", "content": message, "timestamp": datetime.now()}])

    # Process through chat session
    chat_session.insert([{"user_message": message, "timestamp": datetime.now()}])

    # Get response
    result = (
        chat_session.select(chat_session.assistant_response)
        .where(chat_session.user_message == message)
        .collect()
    )

    response = result["assistant_response"][0]

    # Store assistant response in memory
    memory.insert(
        [{"role": "assistant", "content": response, "timestamp": datetime.now()}]
    )

    return response


# Lets test it!!
responses = [
    chat("Hi! My name is Alice."),
    chat("What's the weather like today?"),
    chat("Thanks! Can you remember my name?"),
    chat("What was the first thing I asked you about?"),
]

# Print the conversation
for i, response in enumerate(responses, 1):
    print(f"\nExchange {i}:")
    print(f"Bot: {response}")
