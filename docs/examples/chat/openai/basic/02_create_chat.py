from config import DIRECTORY, OPENAI_MODEL

import pixeltable as pxt
from pixeltable.functions import openai

# Create the table
print('Creating conversations table...')
conversations = pxt.create_table(
    path_str=f'{DIRECTORY}.conversations',
    schema_or_df={'prompt': pxt.String},
    if_exists='ignore',
)

# Create the message column
conversations.add_computed_column(messages=[{'role': 'user', 'content': conversations.prompt}])

# Create the OpenAI response column
conversations.add_computed_column(
    response=openai.chat_completions(
        messages=conversations.messages,
        model=OPENAI_MODEL,
    )
)

# Create the answer column
conversations.add_computed_column(answer=conversations.response.choices[0].message.content)
print('Setup complete!')
