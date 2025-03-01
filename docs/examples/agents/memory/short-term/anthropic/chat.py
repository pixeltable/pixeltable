from datetime import datetime

from config import DIRECTORY

import pixeltable as pxt

memory = pxt.get_table(f'{DIRECTORY}.memory')
agent = pxt.get_table(f'{DIRECTORY}.chat_session')


def chat(message: str):
    # Store user message in memory
    memory.insert([{'role': 'user', 'content': message, 'timestamp': datetime.now()}])

    # Process through chat session
    agent.insert([{'user_message': message, 'timestamp': datetime.now()}])

    # Get response
    result = agent.select(agent.assistant_response).where(agent.user_message == message).collect()

    response = result['assistant_response'][0]

    # Store assistant response in memory
    memory.insert([{'role': 'assistant', 'content': response, 'timestamp': datetime.now()}])
    return response


# Lets test it!!
responses = [
    chat('Hi! My name is Alice.'),
    chat("What's the weather like today?"),
    chat('Thanks! Can you remember my name?'),
    chat('What was the first thing I asked you about?'),
]

# Print the conversation
for i, response in enumerate(responses, 1):
    print(f'\nExchange {i}:')
    print(f'Bot: {response}')
