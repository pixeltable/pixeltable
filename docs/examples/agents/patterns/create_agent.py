from typing import Dict, List

from config import DIRECTORY, OPENAI_MODEL

import pixeltable as pxt



def create_agent():

    # Create fresh environment (optional)
    pxt.drop_dir(DIRECTORY, force=True)
    pxt.create_dir(DIRECTORY, if_exists='ignore')

    # Create messages table to store conversation history
    messages = pxt.create_table(
        path_str=f'{DIRECTORY}.messages',
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


    # Query to retrieve recent messages
    @pxt.query
    def get_recent_messages():
        return (
            messages.order_by(messages.timestamp).select(role=messages.role, content=messages.content).limit(10)
        )  # Get last 10 messages for context window


    @pxt.udf
    def create_messages(past_context: List[Dict], current_message: str) -> List[Dict]:
        """Create messages list with system prompt, messages context and new message"""
        messages = [
            {
                'role': 'system',
                'content': "pretend to be new to programming",
            }
        ]

        # Add conversation history from messages
        messages.extend([{'role': msg['role'], 'content': msg['content']} for msg in past_context])

        # Add current message
        messages.append({'role': 'user', 'content': current_message})

        return messages


    # Add computed columns for response generation pipeline
    chat_session.add_computed_column(messages_context=get_recent_messages())
    chat_session.add_computed_column(prompt=create_messages(chat_session.messages_context, chat_session.user_message))
    chat_session.add_computed_column(
        llm_response=pxt.functions.openai.chat_completions(messages=chat_session.prompt, model=OPENAI_MODEL)
    )
    chat_session.add_computed_column(assistant_response=chat_session.llm_response.choices[0].message.content)
