from typing import Dict, List

from config import DIRECTORY, OPENAI_MODEL

import pixeltable as pxt

# Create fresh environment (optional)
pxt.drop_dir(DIRECTORY, force=True)
pxt.create_dir(DIRECTORY, if_exists='ignore')

# Create memory table to store conversation history
memory = pxt.create_table(
    path_str=f'{DIRECTORY}.memory',
    schema_or_df={
        'role': pxt.String,  # 'user' or 'assistant'
        'content': pxt.String,  # message content
        'timestamp': pxt.Timestamp,  # for ordering and context window
    },
    if_exists='ignore',
)

# Create chat_session table for processing responses with context
chat_session = pxt.create_table(
    path_str=f'{DIRECTORY}.chat_session',
    schema_or_df={'user_message': pxt.String, 'timestamp': pxt.Timestamp},
    if_exists='ignore',
)


# Query to retrieve recent memory
@pxt.query
def get_recent_memory():
    return (
        memory.order_by(memory.timestamp).select(role=memory.role, content=memory.content).limit(10)
    )  # Get last 10 messages for context window


@pxt.udf
def create_messages(past_context: List[Dict], current_message: str) -> List[Dict]:
    """Create messages list with system prompt, memory context and new message"""
    messages = [
        {
            'role': 'system',
            'content': """You are a chatbot with memory capabilities. 
        Use the conversation history to provide contextual and informed responses.
        Remember previous interactions and refer back to them when relevant.""",
        }
    ]

    # Add conversation history from memory
    messages.extend([{'role': msg['role'], 'content': msg['content']} for msg in past_context])

    # Add current message
    messages.append({'role': 'user', 'content': current_message})

    return messages


# Add computed columns for response generation pipeline
chat_session.add_computed_column(memory_context=get_recent_memory())
chat_session.add_computed_column(prompt=create_messages(chat_session.memory_context, chat_session.user_message))
chat_session.add_computed_column(
    llm_response=pxt.functions.openai.chat_completions(messages=chat_session.prompt, model=OPENAI_MODEL)
)
chat_session.add_computed_column(assistant_response=chat_session.llm_response.choices[0].message.content)
