from config import MODEL, DIRECTORY

import pixeltable as pxt
from pixeltable.functions.fireworks import chat_completions

# Create a fresh directory
pxt.drop_dir(DIRECTORY, force=True)
pxt.create_dir(DIRECTORY, if_exists='ignore')

# Create the table
print('Creating conversations table...')
conversations = pxt.create_table(
    path_str=f'{DIRECTORY}.conversations',
    schema_or_df={'prompt': pxt.String},
    if_exists='ignore',
)

# Create the message column
conversations.add_computed_column(messages=[{'role': 'user', 'content': conversations.prompt}])

# Create the Fireworks response column
conversations.add_computed_column(
    response=chat_completions(
        messages=conversations.messages,
        model=MODEL,
    )
)

# Create the answer column
conversations.add_computed_column(answer=conversations.response.choices[0].message.content)
print('Setup complete!')
