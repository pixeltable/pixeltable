import pixeltable as pxt
from pixeltable.functions.fireworks import chat_completions

DIRECTORY = 'fireworks'
MODEL = 'accounts/fireworks/models/deepseek-r1'
RECREATE = False

table_name = f'{DIRECTORY}.chatbot'

# Drop the table if we want to recreate it
if RECREATE:
    pxt.drop_table(table_name)

# Create the table if it doesn't exist
if table_name not in pxt.list_tables():
    # Create a fresh directory
    pxt.create_dir(DIRECTORY, if_exists='ignore')

    # Create the table
    conversations = pxt.create_table(
        path_str=table_name, schema_or_df={'prompt': pxt.String}, if_exists='ignore'
    )

    # Create the message column
    conversations.add_computed_column(messages=[{'role': 'user', 'content': conversations.prompt}])

    # Create the OpenAI response column
    conversations.add_computed_column(response=chat_completions(messages=conversations.messages, model=MODEL))

    # Create the answer column
    conversations.add_computed_column(answer=conversations.response.choices[0].message.content)

else:
    # Load the persistent table
    conversations = pxt.get_table(table_name)

# # Lets test it!!
conversations.insert([{'prompt': 'Tell me a 4 word joke.'}])

# Print the results
print(conversations.select(conversations.prompt, conversations.answer).collect())
